"""LocalStore module."""

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple, Type, Union

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound
from diffsync.store import BaseStore

if TYPE_CHECKING:
    from diffsync import DiffSyncModel


class LocalStore(BaseStore):
    """LocalStore class.

    The raw key/value half of this backend (``get_raw`` / ``set_raw`` /
    ``get_or_load_raw``) implements the in-process side of the three tradeoffs
    documented in :mod:`diffsync.store.tracing`:

    1. Blobs are opaque bytes here; pickle-vs-JSON detection happens in the
       caller, which tries pickle first and falls back to JSON for foreign keys.
    2. Index/root/node keys share one dict but use disjoint suffixes (``:idx``,
       ``:root:...`` and ``:node:...``); the fallback ``scan_keys_raw`` strips
       non-node keys so it cannot confuse the tree root with children.
    3. Parent and child missing the store at the same time are deduplicated with
       a process-local in-flight registry guarded by a lock: concurrent callers
       for the *same* key share one source fetch, and the populated value is
       written exactly once (first writer wins, duplicate write skipped).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init method for LocalStore."""
        super().__init__(*args, **kwargs)

        self._data: Dict = defaultdict(dict)
        # Raw blob store for the Diff tree cache, kept apart from ``_data``
        # which holds live DiffSyncModel instances keyed by (modelname, uid).
        self._raw: Dict[str, bytes] = {}
        # Simple in-memory sets backing ``add_index_raw`` / ``index_members_raw``.
        self._raw_index: Dict[str, Set[str]] = defaultdict(set)
        # One lock serializes raw writes and protects the in-flight registry;
        # readers of a key another thread is loading wait on ``_raw_events``.
        self._raw_lock = threading.Lock()
        self._raw_events: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # Unified raw key/value API (see BaseStore) - LocalStore implementation.
    # ------------------------------------------------------------------

    def get_raw(self, key: str) -> Union[bytes, None]:
        """Return the raw blob at ``key`` or ``None``.

        The hit/miss outcome is traced; cross-protocol classification is added
        by the caller once it has decoded the blob (tradeoff #1).
        """
        value = self._raw.get(key)
        if value is None:
            self._trace(op="get", key=key, result="miss")
        else:
            self._trace(op="get", key=key, result="hit", protocol="raw")
        return value

    def set_raw(self, key: str, value: bytes, *, nx: bool = False) -> bool:
        """Write ``value`` at ``key``; with ``nx=True`` first writer wins."""
        with self._raw_lock:
            if nx and key in self._raw:
                self._trace(
                    op="set",
                    key=key,
                    result="write-skipped",
                    detail="key already present; first writer wins, no overwrite",
                )
                return False
            self._raw[key] = value
        self._trace(op="set", key=key, result="written")
        return True

    def get_or_load_raw(self, key: str, loader: Callable[[], bytes]) -> Tuple[bytes, bool]:
        """Get ``key`` or run a single shared ``loader`` on a concurrent miss.

        Tradeoff #3, LocalStore side: because every caller lives in this
        process, the store owns same-key dedup. The first miss registers an
        in-flight event, runs the source fetch and writes once; concurrent
        misses for the same key block on that event and then read the shared
        value (their ``loaded`` flag is False), so a parent and its child
        loading concurrently never trigger two fetches or two writes.
        """
        with self._raw_lock:
            value = self._raw.get(key)
            if value is not None:
                self._trace(op="get", key=key, result="hit", protocol="raw")
                return value, False
            event = self._raw_events.get(key)
            if event is None:
                # First concurrent miss: this thread owns the source fetch.
                event = threading.Event()
                self._raw_events[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            self._trace(op="get", key=key, result="miss", detail="waiting on in-flight loader")
            event.wait()
            value = self._raw[key]
            self._trace(op="get", key=key, result="hit", protocol="raw", detail="shared loader result")
            return value, False

        try:
            self._trace(op="get", key=key, result="miss", detail="back-to-source")
            loaded_value = loader()
            with self._raw_lock:
                self._raw[key] = loaded_value
            self._trace(op="set", key=key, result="written", detail="dedup owner write")
            return loaded_value, True
        finally:
            with self._raw_lock:
                self._raw_events.pop(key, None)
            event.set()

    def scan_keys_raw(self, pattern: str) -> List[str]:
        """Fallback O(N) glob scan (tradeoff #2), used only on index failure.

        The ``*`` wildcard is translated from the Redis-style glob. Root and
        index keys live outside the scanned ``node:`` segment by naming, and
        are additionally filtered out here so a scan for children can never
        return the tree root itself.
        """
        prefix, _, suffix = pattern.partition("*")
        matches = [
            key
            for key in self._raw
            if key.startswith(prefix) and key.endswith(suffix) and ":node:" in key
        ]
        self._trace(op="scan", key=pattern, result="scan", detail=f"{len(matches)} node key(s)")
        return matches

    def add_index_raw(self, index_key: str, member: str) -> None:
        """Add a node key to the maintained index set."""
        self._raw_index[index_key].add(member)

    def index_members_raw(self, index_key: str) -> Union[List[str], None]:
        """Return the indexed node keys, or ``None`` when no index exists."""
        members = self._raw_index.get(index_key)
        if not members:
            return None
        return sorted(members)

    def get_all_model_names(self) -> Set[str]:
        """Get all the model names stored.

        Return:
            Set of all the model names.
        """
        return set(self._data.keys())

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

        if uid not in self._data[modelname]:
            raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
        return self._data[modelname][uid]

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

        return list(self._data[modelname].values())

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
            if uid not in self._data[modelname]:
                raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
            results.append(self._data[modelname][uid])
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

        existing_obj = self._data[modelname].get(uid)
        if existing_obj:
            if existing_obj is not obj:
                raise ObjectAlreadyExists(f"Object {uid} already present", obj)
            # Return so we don't have to change anything on the existing object and underlying data
            return

        if not obj.adapter:
            obj.adapter = self.adapter

        self._data[modelname][uid] = obj

    def update(self, *, obj: "DiffSyncModel") -> None:
        """Update a DiffSyncModel object to the store.

        Args:
            obj: Object to update
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()

        existing_obj = self._data[modelname].get(uid)
        if existing_obj is obj:
            return

        self._data[modelname][uid] = obj

    def remove_item(self, modelname: str, uid: str) -> None:
        """Remove one item from store."""
        if uid not in self._data[modelname]:
            raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
        del self._data[modelname][uid]

    def count(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"], None] = None) -> int:
        """Returns the number of elements of a specific model, or all elements in the store if unspecified."""
        if not model:
            return sum(len(entries) for entries in self._data.values())

        if isinstance(model, str):
            modelname = model
        else:
            modelname = model.get_type()
        return len(self._data[modelname])
