"""BaseStore module."""

import json
import uuid
from dataclasses import dataclass
from itertools import count
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union, cast
from urllib.parse import quote, unquote

import structlog  # type: ignore

from diffsync.diff import Diff
from diffsync.exceptions import ObjectNotFound

if TYPE_CHECKING:
    from diffsync import Adapter, DiffSyncModel
    from diffsync.diff import DiffElement


DiffProtocol = Literal["json", "pickle"]


@dataclass
class StoreEvent:
    """One observable storage interaction."""

    sequence: int
    operation: str
    scope: str
    key: str
    result: str
    protocol: Optional[str] = None
    node: Optional[str] = None
    detail: Optional[str] = None


def _diff_protocol_payload(value: Any, protocol: DiffProtocol) -> Dict[str, Any]:
    """Wrap a diff payload with its serialization protocol."""
    return {"__diffsync_diff_protocol__": protocol, "data": value}


def encode_diff_payload(value: Any, protocol: DiffProtocol = "json") -> Union[str, bytes]:
    """Encode a diff payload using JSON first or pickle when explicitly requested."""
    payload = _diff_protocol_payload(value, protocol)
    if protocol == "json":
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return pickle_dumps(payload)


def decode_diff_payload(raw: Union[str, bytes]) -> Tuple[Any, DiffProtocol]:
    """Decode a diff payload, trying JSON before falling back to pickle.

    JSON is the portable default. Pickle remains a fallback for keys written by an older
    pickle producer; both paths require the envelope written by these Store APIs.
    """
    try:
        if isinstance(raw, bytes):
            decoded = json.loads(raw.decode("utf-8"))
        else:
            decoded = json.loads(raw)
        actual_protocol = "json"
    except (UnicodeDecodeError, json.JSONDecodeError):
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("pickle payload must be bytes")
        decoded = pickle_loads(raw)  # noqa: S301
        actual_protocol = "pickle"

    if isinstance(decoded, dict) and "__diffsync_diff_protocol__" in decoded:
        actual_protocol = decoded["__diffsync_diff_protocol__"]
        decoded = decoded["data"]
    return decoded, cast(DiffProtocol, actual_protocol)


def encode_diff_node_id(*parts: Any) -> str:
    """Build a collision-resistant slash-delimited diff node identifier."""
    return "/".join(quote(str(part), safe="") for part in parts)


def decode_diff_node_id(node_id: str) -> List[str]:
    """Split a diff node identifier into its quoted path parts."""
    return [unquote(part) for part in node_id.split("/") if part]


class BaseStore:
    """Reference store to be implemented in different backends."""

    def __init__(
        self,  # pylint: disable=unused-argument
        *args: Any,  # pylint: disable=unused-argument
        adapter: Optional["Adapter"] = None,
        name: str = "",
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Init method for BaseStore."""
        self.adapter = adapter
        self.name = name or self.__class__.__name__
        self.store_id = str(uuid.uuid4())
        self.trace_events: List[StoreEvent] = []
        self._trace_sequence = count(1)
        self.last_diff_rebuild_order: List[str] = []
        self._log = structlog.get_logger().new(store=self)

    def __str__(self) -> str:
        """Render store name."""
        return self.name

    def trace_clear(self) -> None:
        """Remove all observed storage events."""
        self.trace_events = []

    def trace_summary(self) -> Dict[str, object]:
        """Return trace counters and hit rate."""
        gets = [event for event in self.trace_events if event.operation == "get"]
        hits = len([event for event in gets if event.result == "hit"])
        misses = len([event for event in gets if event.result in ("miss", "backfill")])
        sets = len([event for event in self.trace_events if event.operation == "set"])
        backfills = len([event for event in gets if event.result == "backfill"])
        total = hits + misses
        return {
            "gets": total,
            "hits": hits,
            "misses": misses,
            "sets": sets,
            "backfills": backfills,
            "hit_rate": (hits / total) if total else 0.0,
        }

    def _trace(
        self,
        *,
        operation: str,
        scope: str,
        key: str,
        result: str,
        protocol: Optional[str] = None,
        node: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Record one storage event for traces and tests."""
        self.trace_events.append(
            StoreEvent(
                sequence=next(self._trace_sequence),
                operation=operation,
                scope=scope,
                key=key,
                result=result,
                protocol=protocol,
                node=node,
                detail=detail,
            )
        )

    def trace_key_events(self, key: str) -> List[StoreEvent]:
        """Return all observed events for one key in call order."""
        return [event for event in self.trace_events if event.key == key]

    def _diff_tree_prefix(self, tree: str) -> str:
        """Get the backend-specific prefix for a diff tree."""
        return f"diff:{self.store_id}:{quote(tree, safe='')}"

    def _diff_root_key(self, tree: str) -> str:
        """Get root metadata key outside the scannable child-node prefix."""
        return f"{self._diff_tree_prefix(tree)}:root"

    def _diff_index_key(self, tree: str) -> str:
        """Get the child-node index key outside the scannable child-node prefix."""
        return f"{self._diff_tree_prefix(tree)}:index"

    def _diff_node_key(self, tree: str, node_id: str) -> str:
        """Get a child-node key under an explicit node segment."""
        return f"{self._diff_tree_prefix(tree)}:node:{node_id}"

    def _diff_node_prefix(self, tree: str) -> str:
        """Get the prefix that contains only child nodes."""
        return f"{self._diff_tree_prefix(tree)}:node:*"

    def _backend_raw_get(self, key: str) -> Optional[Union[str, bytes]]:
        """Get one encoded value from the concrete backend."""
        raise NotImplementedError

    def _backend_raw_set(self, key: str, value: Union[str, bytes], *, nx: bool = False) -> bool:
        """Set one encoded value in the concrete backend."""
        raise NotImplementedError

    def _backend_raw_delete(self, key: str) -> None:
        """Delete one encoded value from the concrete backend."""
        raise NotImplementedError

    def _backend_raw_scan_prefix(self, prefix: str) -> List[str]:
        """List keys under a concrete backend prefix."""
        raise NotImplementedError

    def _raw_get(
        self, key: str, *, scope: str = "diff-node", node: Optional[str] = None
    ) -> Optional[Tuple[Any, DiffProtocol]]:
        """Get and decode a diff payload while recording the lookup."""
        raw = self._backend_raw_get(key)
        if raw is None:
            self._trace(operation="get", scope=scope, key=key, result="miss", node=node)
            return None
        value, protocol = decode_diff_payload(raw)
        self._trace(
            operation="get",
            scope=scope,
            key=key,
            result="hit",
            protocol=protocol,
            node=node,
        )
        return value, protocol

    def _raw_set(
        self,
        key: str,
        value: Any,
        *,
        protocol: DiffProtocol = "json",
        scope: str = "diff-node",
        node: Optional[str] = None,
        nx: bool = False,
    ) -> bool:
        """Encode and set a diff payload while recording the write."""
        created = self._backend_raw_set(key, encode_diff_payload(value, protocol), nx=nx)
        self._trace(
            operation="set",
            scope=scope,
            key=key,
            result="set" if created else "duplicate",
            protocol=protocol,
            node=node,
        )
        return created

    def _get_or_set_raw(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        scope: str = "diff-node",
        node: Optional[str] = None,
        protocol: DiffProtocol = "json",
    ) -> Tuple[Any, DiffProtocol, str, bool]:
        """Get a payload or call factory once on a miss.

        Concrete backends override this to own same-key request deduplication.
        Parent and child are distinct keys, so both may miss and backfill together;
        duplicate writes for one key are not allowed to overwrite the first value.
        """
        existing = self._raw_get(key, scope=scope, node=node)
        if existing is not None:
            value, found_protocol = existing
            return value, found_protocol, "hit", False

        value = factory()
        self._raw_set(key, value, protocol=protocol, scope=scope, node=node)
        self._trace(operation="get", scope=scope, key=key, result="backfill", protocol=protocol, node=node)
        return value, protocol, "backfill", True

    @staticmethod
    def _element_to_payload(element: "DiffElement") -> Dict[str, Any]:
        """Serialize one DiffElement without embedding its child tree."""
        return {
            "kind": "element",
            "type": element.type,
            "name": element.name,
            "keys": element.keys,
            "source_name": element.source_name,
            "dest_name": element.dest_name,
            "source_attrs": element.source_attrs,
            "dest_attrs": element.dest_attrs,
            "groups": [
                {
                    "type": child_type,
                    "names": list(element.child_diff.children[child_type].keys()),
                }
                for child_type in element.child_diff.groups()
            ],
        }

    @staticmethod
    def _payload_to_element(payload: Dict[str, Any], diff_class: Type["Diff"]) -> "DiffElement":
        """Recreate one DiffElement from its payload."""
        from diffsync.diff import DiffElement

        element = DiffElement(
            payload["type"],
            payload["name"],
            payload["keys"],
            source_name=payload["source_name"],
            dest_name=payload["dest_name"],
            diff_class=diff_class,
        )
        element.add_attrs(source=payload.get("source_attrs"), dest=payload.get("dest_attrs"))
        return element

    @staticmethod
    def _diff_to_payload(diff: "Diff") -> Dict[str, Any]:
        """Serialize root group order."""
        return {
            "kind": "diff",
            "models_processed": diff.models_processed,
            "groups": [
                {"type": group_type, "names": list(diff.children[group_type].keys())}
                for group_type in diff.groups()
            ],
        }

    @staticmethod
    def _payload_to_diff(payload: Optional[Dict[str, Any]] = None) -> "Diff":
        """Recreate an empty root Diff."""
        from diffsync.diff import Diff

        diff = Diff()
        if payload:
            diff.models_processed = int(payload.get("models_processed", 0))
        return diff

    def dump_diff(self, diff: "Diff", *, tree: str = "default", protocol: DiffProtocol = "json") -> None:
        """Persist child nodes first, root next, and the preorder index last.

        The index makes normal reconstruction O(nodes) direct gets. A root and child
        prefix cannot safely share a trailing wildcard, so node keys use an explicit
        `:node:` segment and root/index sit beside it. If the index is later lost,
        the first reader pays one `:node:*` scan and rewrites the recovered index.
        """
        preorder: List[str] = []
        postorder: List[Tuple[str, Dict[str, Any]]] = []

        def collect(element: "DiffElement", node_id: str) -> None:
            preorder.append(node_id)
            payload = self._element_to_payload(element)
            for group_payload in payload["groups"]:
                for child_name in group_payload["names"]:
                    child = element.child_diff.children[group_payload["type"]][child_name]
                    child_id = f"{node_id}/{encode_diff_node_id(group_payload['type'], child.name)}"
                    collect(child, child_id)
            postorder.append((node_id, payload))

        for group in diff.groups():
            for child in diff.children[group].values():
                collect(child, encode_diff_node_id(group, child.name))

        for node_id, payload in postorder:
            self._raw_set(
                self._diff_node_key(tree, node_id),
                payload,
                protocol=protocol,
                scope="diff-node",
                node=node_id,
            )

        self._raw_set(
            self._diff_root_key(tree),
            self._diff_to_payload(diff),
            protocol=protocol,
            scope="diff-root",
            node="<root>",
        )
        self._raw_set(
            self._diff_index_key(tree),
            {"kind": "diff-index", "order": preorder},
            protocol=protocol,
            scope="diff-index",
            node="<index>",
        )

    def load_diff(self, *, tree: str = "default", diff_class: Optional[Type["Diff"]] = None) -> "Diff":
        """Rebuild a Diff in the stored preorder, repairing a missing index once."""
        root_key = self._diff_root_key(tree)
        index_key = self._diff_index_key(tree)

        root = self._raw_get(root_key, scope="diff-root", node="<root>")
        if root is None:
            raise ObjectNotFound(f"Diff root {root_key} not present in {str(self)}")
        root_payload, _ = root

        index = self._raw_get(index_key, scope="diff-index", node="<index>")
        if index is None:
            self._repair_diff_index(tree, root_payload)
            repaired_index_raw = self._backend_raw_get(index_key)
            if repaired_index_raw is None:
                raise ObjectNotFound(f"Diff index {index_key} could not be repaired in {str(self)}")
            index_payload, _ = decode_diff_payload(repaired_index_raw)
        else:
            index_payload, _ = index

        nodes: Dict[str, "DiffElement"] = {}
        rebuild_order: List[str] = []
        def build_element(node_id: str) -> "DiffElement":
            if node_id in nodes:
                return nodes[node_id]
            rebuild_order.append(node_id)
            node = self._raw_get(self._diff_node_key(tree, node_id), scope="diff-node", node=node_id)
            if node is None:
                raise ObjectNotFound(f"Diff node {node_id} not present in {str(self)}")
            payload, _ = node
            element = self._payload_to_element(payload, diff_class if diff_class is not None else Diff)
            for group_payload in payload.get("groups", []):
                for name in group_payload["names"]:
                    child_id = f"{node_id}/{encode_diff_node_id(group_payload['type'], name)}"
                    element.add_child(build_element(child_id))
            nodes[node_id] = element
            return element

        root_class = diff_class if diff_class is not None else Diff
        diff = root_class()
        diff.models_processed = int(root_payload.get("models_processed", 0))
        for group_payload in root_payload.get("groups", []):
            for name in group_payload["names"]:
                node_id = encode_diff_node_id(group_payload["type"], name)
                diff.add(build_element(node_id))

        self.last_diff_rebuild_order = rebuild_order
        return diff

    def _repair_diff_index(self, tree: str, root_payload: Dict[str, Any]) -> None:
        """Rebuild and rewrite the index with one bounded node-prefix scan.

        The first reader after index loss accepts the scan cost once. Existing
        payloads are still read individually, because scan only identifies keys and
        must not silently bypass hit/miss observability or protocol fallback.
        """
        expected: Set[str] = set()

        def discover(node_id: str, order: List[str]) -> None:
            expected.add(node_id)
            order.append(node_id)
            node = self._raw_get(self._diff_node_key(tree, node_id), scope="diff-node", node=node_id)
            if node is None:
                return
            payload, _ = node
            for group_payload in payload.get("groups", []):
                for name in group_payload["names"]:
                    discover(f"{node_id}/{encode_diff_node_id(group_payload['type'], name)}", order)

        ordered_nodes: List[str] = []
        for group_payload in root_payload.get("groups", []):
            for name in group_payload["names"]:
                discover(encode_diff_node_id(group_payload["type"], name), ordered_nodes)

        scanned = set(self._backend_raw_scan_prefix(self._diff_node_prefix(tree)))
        orphaned = sorted(scanned - {self._diff_node_key(tree, node_id) for node_id in expected})
        self._trace(
            operation="scan",
            scope="diff-repair",
            key=self._diff_node_prefix(tree),
            result="index-repaired",
            detail=f"orphaned={len(orphaned)}",
        )
        self._raw_set(
            self._diff_index_key(tree),
            {"kind": "diff-index", "order": ordered_nodes},
            scope="diff-index",
            node="<index>",
        )

    def get_or_set_diff_node(
        self,
        *,
        tree: str,
        node_id: str,
        factory: Callable[[], "DiffElement"],
        protocol: DiffProtocol = "json",
    ) -> Tuple["DiffElement", DiffProtocol, str, bool]:
        """Get one DiffElement or call the node-specific source factory on a miss.

        Same-node concurrent requests are collapsed by LocalStore in-process and by
        RedisStore's backend lock. A parent miss and child miss remain independent
        source calls; their first written values are preserved without overwrite.
        """
        def payload_factory() -> Dict[str, Any]:
            return self._element_to_payload(factory())

        payload, found_protocol, result, backfilled = self._get_or_set_raw(
            self._diff_node_key(tree, node_id),
            payload_factory,
            scope="diff-node",
            node=node_id,
            protocol=protocol,
        )
        return self._payload_to_element(payload, Diff), found_protocol, result, backfilled

    def set_diff_node(
        self, *, tree: str, node_id: str, element: "DiffElement", protocol: DiffProtocol = "json"
    ) -> None:
        """Explicitly write one DiffElement node, overwriting intentional updates."""
        self._raw_set(
            self._diff_node_key(tree, node_id),
            self._element_to_payload(element),
            protocol=protocol,
            scope="diff-node",
            node=node_id,
        )

    def remove_diff_node(self, *, tree: str, node_id: str) -> None:
        """Remove one DiffElement node."""
        key = self._diff_node_key(tree, node_id)
        self._backend_raw_delete(key)
        self._trace(operation="delete", scope="diff-node", key=key, result="delete", node=node_id)

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


# Compatibility alias for callers/docs referring to the unified Store API.
Store = BaseStore

