"""LocalStore module."""

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound
from diffsync.store import BaseStore, DiffProtocol, decode_diff_payload

if TYPE_CHECKING:
    from diffsync import DiffSyncModel


class LocalStore(BaseStore):
    """LocalStore class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init method for LocalStore."""
        super().__init__(*args, **kwargs)

        self._data: Dict = defaultdict(dict)
        self._raw_data: Dict[str, Union[str, bytes]] = {}
        self._raw_lock = threading.RLock()
        self._in_flight: Dict[str, threading.Event] = {}

    @staticmethod
    def _model_key(modelname: str, uid: str) -> str:
        """Build the observed key for a locally stored model object."""
        return f"local:model:{modelname}:{uid}"

    def _backend_raw_get(self, key: str) -> Union[str, bytes, None]:
        """Read one raw local key."""
        with self._raw_lock:
            return self._raw_data.get(key)

    def _backend_raw_set(self, key: str, value: Union[str, bytes], *, nx: bool = False) -> bool:
        """Write one raw local key without overwrite when nx is requested."""
        with self._raw_lock:
            if nx and key in self._raw_data:
                return False
            self._raw_data[key] = value
            return True

    def _backend_raw_delete(self, key: str) -> None:
        """Delete one raw local key."""
        with self._raw_lock:
            self._raw_data.pop(key, None)

    def _backend_raw_scan_prefix(self, prefix: str) -> List[str]:
        """List raw local keys below a glob prefix."""
        literal_prefix = prefix[:-1] if prefix.endswith("*") else prefix
        with self._raw_lock:
            return [key for key in self._raw_data if key.startswith(literal_prefix)]

    def _get_or_set_raw(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        scope: str = "diff-node",
        node: Optional[str] = None,
        protocol: DiffProtocol = "json",
    ) -> Tuple[Any, DiffProtocol, str, bool]:
        """Deduplicate simultaneous requests for the same in-process key.

        LocalStore owns in-process coordination. Different parent and child keys use
        separate events and backfill together; the winning payload is kept while a
        racing duplicate write resolves to a hit instead of overwriting.
        """
        existing = self._raw_get(key, scope=scope, node=node)
        if existing is not None:
            value, found_protocol = existing
            return value, found_protocol, "hit", False

        with self._raw_lock:
            if key in self._raw_data:
                value, found_protocol = decode_diff_payload(self._raw_data[key])
                self._trace(
                    operation="get",
                    scope=scope,
                    key=key,
                    result="hit",
                    protocol=found_protocol,
                    node=node,
                )
                return value, found_protocol, "hit", False
            if key in self._in_flight:
                ready = self._in_flight[key]
                owner = False
            else:
                ready = threading.Event()
                self._in_flight[key] = ready
                owner = True

        if not owner:
            ready.wait()
            existing_after_wait = self._raw_get(key, scope=scope, node=node)
            if existing_after_wait is None:
                raise ObjectNotFound(f"{key} disappeared after concurrent backfill")
            value, found_protocol = existing_after_wait
            return value, found_protocol, "hit", False

        try:
            value = factory()
            self._raw_set(key, value, protocol=protocol, scope=scope, node=node, nx=True)
            self._trace(
                operation="get",
                scope=scope,
                key=key,
                result="backfill",
                protocol=protocol,
                node=node,
            )
            return value, protocol, "backfill", True
        finally:
            with self._raw_lock:
                self._in_flight.pop(key, None)
                ready.set()

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

        key = self._model_key(modelname, uid)
        if uid not in self._data[modelname]:
            self._trace(operation="get", scope="model", key=key, result="miss")
            raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
        self._trace(operation="get", scope="model", key=key, result="hit")
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
        key = self._model_key(modelname, uid)

        existing_obj = self._data[modelname].get(uid)
        if existing_obj:
            if existing_obj is not obj:
                self._trace(operation="set", scope="model", key=key, result="duplicate")
                raise ObjectAlreadyExists(f"Object {uid} already present", obj)
            # Return so we don't have to change anything on the existing object and underlying data
            self._trace(operation="set", scope="model", key=key, result="duplicate")
            return

        if not obj.adapter:
            obj.adapter = self.adapter

        self._data[modelname][uid] = obj
        self._trace(operation="set", scope="model", key=key, result="set")

    def update(self, *, obj: "DiffSyncModel") -> None:
        """Update a DiffSyncModel object to the store.

        Args:
            obj: Object to update
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()
        key = self._model_key(modelname, uid)

        existing_obj = self._data[modelname].get(uid)
        if existing_obj is obj:
            self._trace(operation="set", scope="model", key=key, result="duplicate")
            return

        self._data[modelname][uid] = obj
        self._trace(operation="set", scope="model", key=key, result="overwrite")

    def remove_item(self, modelname: str, uid: str) -> None:
        """Remove one item from store."""
        key = self._model_key(modelname, uid)
        if uid not in self._data[modelname]:
            self._trace(operation="delete", scope="model", key=key, result="miss")
            raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
        del self._data[modelname][uid]
        self._trace(operation="delete", scope="model", key=key, result="delete")

    def count(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"], None] = None) -> int:
        """Returns the number of elements of a specific model, or all elements in the store if unspecified."""
        if not model:
            return sum(len(entries) for entries in self._data.values())

        if isinstance(model, str):
            modelname = model
        else:
            modelname = model.get_type()
        return len(self._data[modelname])
