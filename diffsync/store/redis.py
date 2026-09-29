"""RedisStore module."""

import copy
import uuid
from pickle import dumps, loads  # nosec
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

try:
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError as ierr:
    print("Redis is not installed. Have you installed diffsync with redis extra? `pip install diffsync[redis]`")
    raise ierr

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound, ObjectStoreException
from diffsync.store import BaseStore

if TYPE_CHECKING:
    from diffsync import DiffSyncModel

REDIS_DIFFSYNC_ROOT_LABEL = "diffsync"


class RedisStore(BaseStore):
    """RedisStore class.

    Raw key/value operations on this backend implement the Redis side of the
    three tradeoffs in :mod:`diffsync.store.tracing`:

    1. Values are stored as opaque bytes; deserialization in the caller tries
       pickle first and degrades to JSON when the key was written by a
       JSON-only peer.
    2. Key layout keeps ``:idx`` / ``:root:`` outside the ``:node:`` prefix,
       so ``SCAN MATCH ...:node:*`` only ever enumerates children. When the
       index is missing/stale the *reader* performs the one-off fallback scan
       and repairs the index; writes are not blocked waiting for the index.
    3. This backend cannot see another process' in-flight registry, so
       parent+child concurrent misses are deduped on the server with
       ``SET key value NX``: redundant source fetches may still happen, but
       only the first writer's value survives - duplicate writes are skipped
       and never overwrite, matching LocalStore's observable contract.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        store_id: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6379,
        url: Optional[str] = None,
        db: int = 0,
        client: Any = None,
        **kwargs: Any,
    ):
        """Init method for RedisStore."""
        super().__init__(*args, **kwargs)

        if url and host and port:
            raise ValueError("'url' and 'host' arguments can't be specified together.")

        try:
            if client is not None:
                # Injected client (real Redis connection or the in-memory
                # FakeRedis used by tests/tracing); bypass host/url discovery.
                self._store = client
            elif url:
                self._store = Redis.from_url(url, db=db)
            elif host:
                self._store = Redis(host=host, port=port, db=db)
            else:
                raise RedisConnectionError("Neither 'host' nor 'url' were specified.")

            if not self._store.ping():
                raise RedisConnectionError()
        except RedisConnectionError:
            raise ObjectStoreException("Redis store is unavailable.") from RedisConnectionError

        self._store_id = store_id if store_id else str(uuid.uuid4())

        self._store_label = f"{REDIS_DIFFSYNC_ROOT_LABEL}:{self._store_id}"

    def __str__(self) -> str:
        """Render store name."""
        return f"{self.name} ({self._store_id})"

    def _get_object_from_redis_key(self, key: str) -> "DiffSyncModel":
        """Get the object from Redis key."""
        pickled_object = self._store.get(key)
        if pickled_object:
            obj_result = loads(pickled_object)  # noqa: S301
            obj_result.adapter = self.adapter
            return obj_result
        raise ObjectNotFound(f"{key} not present in Cache")

    def get_all_model_names(self) -> Set[str]:
        """Get all the model names stored.

        Return:
            Set of all the model names.
        """
        # TODO: optimize it
        all_model_names = set()
        for item in self._store.scan_iter(f"{self._store_label}:*"):
            # Model Name is the third item in the Redis key
            # b'diffsync:123:device:device1' -> Model name b'device'
            model_name = item.split(b":")[2].decode()
            all_model_names.add(model_name)

        return all_model_names

    def _get_key_for_object(self, modelname: str, uid: str) -> str:
        return f"{self._store_label}:{modelname}:{uid}"

    def get(
        self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]], identifier: Union[str, Dict]
    ) -> "DiffSyncModel":
        """Get one object from the data store based on its unique id.

        Args:
            model: DiffSyncModel class or instance, or modelname string, that defines the type of the object to retrieve
            identifier: Unique ID of the object to retrieve, or dict of unique identifier keys/values

        Raises:
            ValueError: if obj is a str and identifier is a dict (can't convert dict into a uid str without a model class)
            ObjectNotFound: if the requested object is not present
        """
        object_class, modelname = self._get_object_class_and_model(model)

        uid = self._get_uid(model, object_class, identifier)

        return self._get_object_from_redis_key(self._get_key_for_object(modelname, uid))

    def get_all(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]) -> List["DiffSyncModel"]:
        """Get all objects of a given type.

        Args:
            model: DiffSyncModel class or instance, or modelname string, that defines the type of the objects to retrieve

        Returns:
            List of Object
        """
        if isinstance(model, str):
            modelname = model
        else:
            modelname = model.get_type()

        results: List["DiffSyncModel"] = []
        for key in self._store.scan_iter(f"{self._store_label}:{modelname}:*"):
            results.append(self._get_object_from_redis_key(key))  # type: ignore[arg-type]

        return results

    def get_by_uids(
        self, *, uids: List[str], model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]
    ) -> List["DiffSyncModel"]:
        """Get multiple objects from the store by their unique IDs/Keys and type.

        Args:
            uids: List of unique id / key identifying object in the database.
            model: DiffSyncModel class or instance, or modelname string, that defines the type of the objects to retrieve

        Raises:
            ObjectNotFound: if any of the requested UIDs are not found in the store
        """
        if isinstance(model, str):
            modelname = model
        else:
            modelname = model.get_type()

        results = []
        for uid in uids:
            results.append(self._get_object_from_redis_key(self._get_key_for_object(modelname, uid)))

        return results

    def add(self, *, obj: "DiffSyncModel") -> None:
        """Add a DiffSyncModel object to the store.

        Args:
            obj: Object to store

        Raises:
            ObjectAlreadyExists: if a different object with the same uid is already present.
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()

        # Get existing Object
        object_key = self._get_key_for_object(modelname, uid)

        existing_obj_binary = self._store.get(object_key)
        if existing_obj_binary:
            existing_obj = loads(existing_obj_binary)  # noqa: S301
            existing_obj_dict = existing_obj.dict()

            if existing_obj_dict != obj.dict():
                raise ObjectAlreadyExists(f"Object {uid} already present", obj)

            # Return so we don't have to change anything on the existing object and underlying data
            return

        # Remove the diffsync object before sending to Redis
        obj_copy = copy.copy(obj)
        obj_copy.adapter = None

        self._store.set(object_key, dumps(obj_copy))

    def update(self, *, obj: "DiffSyncModel") -> None:
        """Update a DiffSyncModel object to the store.

        Args:
            obj: Object to update
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()

        object_key = self._get_key_for_object(modelname, uid)
        obj_copy = copy.copy(obj)
        obj_copy.adapter = None

        self._store.set(object_key, dumps(obj_copy))

    def remove_item(self, modelname: str, uid: str) -> None:
        """Remove one item from store."""
        object_key = self._get_key_for_object(modelname, uid)

        if not self._store.exists(object_key):
            raise ObjectNotFound(f"{modelname} {uid} not present in Cache")

        self._store.delete(object_key)

    def count(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"], None] = None) -> int:
        """Returns the number of elements of a specific model, or all elements in the store if unspecified."""
        search_pattern = f"{self._store_label}:*"
        if model is not None:
            if isinstance(model, str):
                modelname = model
            else:
                modelname = model.get_type()
            search_pattern = f"{self._store_label}:{modelname.lower()}:*"

        return len(list(self._store.scan_iter(search_pattern)))

    # ------------------------------------------------------------------
    # Unified raw key/value API (see BaseStore) - RedisStore implementation.
    # ------------------------------------------------------------------

    def get_raw(self, key: str) -> Optional[bytes]:
        """Return the raw blob at ``key`` or ``None`` and trace the outcome."""
        value = self._store.get(key)
        if value is None:
            self._trace(op="get", key=key, result="miss")
            return None
        self._trace(op="get", key=key, result="hit", protocol="raw")
        return value

    def set_raw(self, key: str, value: bytes, *, nx: bool = False) -> bool:
        """Write with optional ``NX`` semantics (first-writer-wins, no overwrite)."""
        written = self._store.set(key, value, nx=nx)
        if written:
            self._trace(op="set", key=key, result="written")
            return True
        self._trace(
            op="set",
            key=key,
            result="write-skipped",
            detail="SET NX lost; first writer wins, no overwrite",
        )
        return False

    def get_or_load_raw(self, key: str, loader: Callable[[], bytes]) -> Tuple[bytes, bool]:
        """Get ``key`` or back-to-source ``loader`` and claim the write with NX.

        Tradeoff #3, RedisStore side: cross-process same-key misses cannot share
        an in-process event, so each concurrent miss may run its own
        ``loader`` (duplicated source work is accepted as the price of no
        shared memory); the duplicate *write* is what is deduped - and only
        on the server, via ``SET NX``. A lost race re-reads the winner's value
        so every caller observes the same first-written blob.
        """
        value = self._store.get(key)
        if value is not None:
            self._trace(op="get", key=key, result="hit", protocol="raw")
            return value, False

        self._trace(op="get", key=key, result="miss", detail="back-to-source")
        loaded_value = loader()
        if self.set_raw(key, loaded_value, nx=True):
            return loaded_value, True
        # Lost the SET NX race to a concurrent first writer: return its blob.
        winner = self._store.get(key)
        self._trace(op="get", key=key, result="hit", protocol="raw", detail="read NX winner")
        return winner, True

    def scan_keys_raw(self, pattern: str) -> List[str]:
        """Fallback O(N) ``SCAN`` traversal used only when the index is stale."""
        matches = []
        for item in self._store.scan_iter(pattern):
            if isinstance(item, bytes):
                item = item.decode()
            # Defensive strip of non-node keys, mirroring LocalStore: root and
            # index keys must never be mistaken for tree children.
            if ":node:" in item:
                matches.append(item)
        self._trace(op="scan", key=pattern, result="scan", detail=f"{len(matches)} node key(s)")
        return matches

    def add_index_raw(self, index_key: str, member: str) -> None:
        """Add a node key to the Redis SET backing the tree index."""
        self._store.sadd(index_key, member)

    def index_members_raw(self, index_key: str) -> Optional[List[str]]:
        """Return the indexed node keys, or ``None`` if the index SET is absent/empty."""
        if not self._store.exists(index_key):
            return None
        members = self._store.smembers(index_key)
        decoded = {member.decode() if isinstance(member, bytes) else member for member in members}
        return sorted(decoded)
