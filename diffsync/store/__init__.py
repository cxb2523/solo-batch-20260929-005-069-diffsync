"""BaseStore module.

Observability note
------------------
Every backend accepts an optional
:class:`~diffsync.store.tracing.Tracer`. The tracer only records what the
backend *already does* (key-level get/set/scan and hit/miss/write-skipped
outcomes); it never changes store semantics. See
:mod:`diffsync.store.tracing` for the three design tradeoffs that bind the
``BaseStore``/``LocalStore``/``RedisStore`` signatures together:

1. pickle-first, JSON-fallback deserialization of foreign-protocol keys;
2. explicit index/root keys kept outside the child ``node:`` prefix, with the
   reader paying a one-off prefix scan when the index is missing/stale;
3. unified ``get_or_load_raw`` signature, with same-key concurrent misses
   deduped in-process by LocalStore and server-side by RedisStore (``SET NX``);
   duplicate writes never overwrite the first writer's value.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import structlog  # type: ignore

from diffsync.exceptions import ObjectNotFound
from diffsync.store.tracing import Tracer

if TYPE_CHECKING:
    from diffsync import Adapter, DiffSyncModel


class BaseStore:
    """Reference store to be implemented in different backends."""

    def __init__(
        self,  # pylint: disable=unused-argument
        *args: Any,  # pylint: disable=unused-argument
        adapter: Optional["Adapter"] = None,
        name: str = "",
        tracer: Optional[Tracer] = None,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Init method for BaseStore."""
        self.adapter = adapter
        self.name = name or self.__class__.__name__
        # Optional observability recorder. Backends MUST route every key-level
        # get/set/scan through ``self._trace`` so that trace.html can reconstruct
        # the exact key sequence and hit/back-to-source classification.
        self.tracer = tracer
        self._log = structlog.get_logger().new(store=self)

    def __str__(self) -> str:
        """Render store name."""
        return self.name

    def _trace(
        self, *, op: str, key: str, result: str, protocol: str = "-", detail: str = ""
    ) -> None:
        """Record an operation on the attached tracer, if any.

        Args:
            op: ``get`` / ``set`` / ``scan`` / ``index``.
            key: Exact store key touched (or glob pattern for a scan).
            result: Outcome category from :mod:`diffsync.store.tracing`.
            protocol: ``pickle`` / ``json`` for payload ops, ``-`` otherwise.
            detail: Optional human-readable context for the trace page.
        """
        if self.tracer is not None:
            self.tracer.record(
                store=str(self), op=op, key=key, result=result, protocol=protocol, detail=detail
            )

    # ------------------------------------------------------------------
    # Unified raw key/value API used by the Diff tree cache.
    #
    # These methods operate on opaque ``bytes`` blobs and flat string keys;
    # they deliberately do NOT know about DiffSyncModel. LocalStore and
    # RedisStore implement the *same* signature so the cache layer above can
    # swap backends without a branch. The contract is fixed here:
    #
    # * ``get_raw`` -> ``Optional[bytes]``; records hit / cross-protocol /
    #   miss and never raises on an absent key (returning ``None`` instead).
    # * ``set_raw(key, value, nx=True)`` -> ``bool``; with ``nx=True`` the
    #   first writer wins and a duplicate write is skipped (recorded as
    #   ``write-skipped``), implementing the no-overwrite half of tradeoff #3.
    # * ``get_or_load_raw`` -> ``(bytes, bool)``; the backend itself owns
    #   same-key miss dedup (in-flight registry for LocalStore, ``SET NX`` for
    #   RedisStore). ``loaded=True`` means *this* caller ran ``loader``;
    #   sharing callers observe ``loaded=False``.
    # * ``scan_keys_raw`` is the *fallback* O(N) traversal behind the index
    #   described in tradeoff #2; hot paths read the index instead.
    # ------------------------------------------------------------------

    def get_raw(self, key: str) -> Optional[bytes]:
        """Return the raw blob stored at ``key``, or ``None`` if absent."""
        raise NotImplementedError

    def set_raw(self, key: str, value: bytes, *, nx: bool = False) -> bool:
        """Store a raw blob.

        Args:
            key: Flat store key.
            value: Serialized blob (pickle or JSON; see tradeoff #1).
            nx: If True, only write when the key is absent (first-writer-wins).

        Returns:
            True if a value was written, False if the key already existed and
            the write was skipped.
        """
        raise NotImplementedError

    def get_or_load_raw(self, key: str, loader: Callable[[], bytes]) -> Tuple[bytes, bool]:
        """Get ``key`` or, on miss, populate it with ``loader()``.

        Same-key concurrent misses are deduplicated by the backend; see the
        class docstring (tradeoff #3).

        Args:
            key: Flat store key.
            loader: Zero-argument callable producing the blob from the source
                of truth ("back-to-source" in the trace).

        Returns:
            ``(blob, loaded)`` where ``loaded`` is True if this caller invoked
            ``loader`` and False if the value was read from the store.
        """
        raise NotImplementedError

    def scan_keys_raw(self, pattern: str) -> List[str]:
        """Return all keys matching ``pattern`` (glob *), as decoded str.

        This is the temporary O(N) traversal used only when the maintained
        index/root keys are missing or stale (tradeoff #2).
        """
        raise NotImplementedError

    def add_index_raw(self, index_key: str, member: str) -> None:
        """Add ``member`` to the set stored at ``index_key`` (create if needed)."""
        raise NotImplementedError

    def index_members_raw(self, index_key: str) -> Optional[List[str]]:
        """Return the index set members, or ``None`` if the index key is absent."""
        raise NotImplementedError

    def get_all_model_names(self) -> Set[str]:
        """Get all the model names stored.

        Return:
            Set of all the model names.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def get_all(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]) -> List["DiffSyncModel"]:
        """Get all objects of a given type.

        Args:
            model: DiffSyncModel class or instance, or modelname string, that defines the type of the objects to retrieve

        Returns:
            List of Object
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def remove_item(self, modelname: str, uid: str) -> None:
        """Remove one item from store."""
        raise NotImplementedError

    def remove(self, *, obj: "DiffSyncModel", remove_children: bool = False) -> None:
        """Remove a DiffSyncModel object from the store.

        Args:
            obj: object to remove
            remove_children: If True, also recursively remove any children of this object

        Raises:
            ObjectNotFound: if the object is not present
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()

        self.remove_item(modelname, uid)

        if obj.adapter:
            obj.adapter = None

        if remove_children:
            for child_type, child_fieldname in obj.get_children_mapping().items():
                for child_id in getattr(obj, child_fieldname):
                    try:
                        child_obj = self.get(model=child_type, identifier=child_id)
                        self.remove(obj=child_obj, remove_children=remove_children)
                    except ObjectNotFound:
                        # Since this is "cleanup" code, log an error and continue, instead of letting the exception raise
                        self._log.error(
                            "Unable to remove child element as it was not found!",
                            child_type=child_type,
                            child_id=child_id,
                            parent_type=modelname,
                            parent_id=uid,
                        )

    def add(self, *, obj: "DiffSyncModel") -> None:
        """Add a DiffSyncModel object to the store.

        Args:
            obj: Object to store

        Raises:
            ObjectAlreadyExists: if a different object with the same uid is already present.
        """
        raise NotImplementedError

    def update(self, *, obj: "DiffSyncModel") -> None:
        """Update a DiffSyncModel object to the store.

        Args:
            obj: Object to update
        """
        raise NotImplementedError

    def count(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"], None] = None) -> int:
        """Returns the number of elements of a specific model, or all elements in the store if not specified."""
        raise NotImplementedError

    def get_or_instantiate(
        self, *, model: Type["DiffSyncModel"], ids: Dict, attrs: Optional[Dict] = None
    ) -> Tuple["DiffSyncModel", bool]:
        """Attempt to get the object with provided identifiers or instantiate it with provided identifiers and attrs.

        Args:
            model: The DiffSyncModel to get or create.
            ids: Identifiers for the DiffSyncModel to get or create with.
            attrs: Attributes when creating an object if it doesn't exist. Defaults to None.

        Returns:
            Provides the existing or new object and whether it was created or not.
        """
        created = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            if not attrs:
                attrs = {}
            obj = model(**ids, **attrs)
            # Add the object to diffsync adapter
            self.add(obj=obj)
            created = True

        return obj, created

    def get_or_add_model_instance(self, obj: "DiffSyncModel") -> Tuple["DiffSyncModel", bool]:
        """Attempt to get the object with provided obj identifiers or instantiate obj.

        Args:
            obj: An obj of the DiffSyncModel to get or add.

        Returns:
            Provides the existing or new object and whether it was added or not.
        """
        model = obj.get_type()
        ids = obj.get_unique_id()

        try:
            return self.get(model=model, identifier=ids), False
        except ObjectNotFound:
            self.add(obj=obj)
            return obj, True

    def update_or_instantiate(
        self, *, model: Type["DiffSyncModel"], ids: Dict, attrs: Dict
    ) -> Tuple["DiffSyncModel", bool]:
        """Attempt to update an existing object with provided ids/attrs or instantiate it with provided identifiers and attrs.

        Args:
            model: The DiffSyncModel to get or create.
            ids: Identifiers for the DiffSyncModel to get or create with.
            attrs: Attributes when creating/updating an object if it doesn't exist. Pass in empty dict, if no specific attrs.

        Returns:
            Provides the existing or new object and whether it was created or not.
        """
        created = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            obj = model(**ids, **attrs)
            # Add the object to diffsync adapter
            self.add(obj=obj)
            created = True

        # Update existing obj with attrs
        for attr, value in attrs.items():
            if getattr(obj, attr) != value:
                setattr(obj, attr, value)

        return obj, created

    def update_or_add_model_instance(self, obj: "DiffSyncModel") -> Tuple["DiffSyncModel", bool]:
        """Attempt to update an existing object with provided ids/attrs or instantiate obj.

        Args:
            obj: An instance of the DiffSyncModel to update or create.

        Returns:
            Provides the existing or new object and whether it was added or not.
        """
        model = obj.get_type()
        ids = obj.get_unique_id()
        attrs = obj.get_attrs()

        added = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            # Add the object to the diffsync instance
            self.add(obj=obj)
            added = True

        # Update existing obj with attrs
        for attr, value in attrs.items():
            setattr(obj, attr, value)

        return obj, added

    def _get_object_class_and_model(
        self, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]
    ) -> Tuple[Union["DiffSyncModel", Type["DiffSyncModel"], None], str]:
        """Get object class and model name for a model."""
        if isinstance(model, str):
            modelname = model
            if not hasattr(self.adapter, model):
                return None, modelname
            object_class = getattr(self.adapter, model)
        else:
            object_class = model
            modelname = model.get_type()

        return object_class, modelname

    @staticmethod
    def _get_uid(
        model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]],
        object_class: Union["DiffSyncModel", Type["DiffSyncModel"], None],
        identifier: Union[str, Dict],
    ) -> str:
        """Get the related uid for a model and an identifier."""
        if isinstance(identifier, str):
            uid = identifier
        elif object_class:
            uid = object_class.create_unique_id(**identifier)
        else:
            raise ValueError(
                f"Invalid args: ({model}, {object_class}, {identifier}): "
                f"either {object_class} should be a class/instance or {identifier} should be a str"
            )
        return uid
