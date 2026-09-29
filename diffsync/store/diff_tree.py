"""Three-level Diff/DiffElement tree cache on top of the unified store API.

This module serializes a :class:`~diffsync.diff.Diff` tree (root ``Diff`` ->
first-level ``DiffElement`` -> second-level child ``DiffElement``) into the
flat key space of a :class:`~diffsync.store.BaseStore` backend and rebuilds it
later. It is the place where the three tradeoffs from
:mod:`diffsync.store.tracing` actually bite together:

1. **Dual protocol.** Every node blob is encoded with one fixed protocol
   (``pickle`` or ``json``), but a rebuild may read keys written by a peer
   using the other one. Decoding *tries pickle first* (it round-trips the
   Diff/DiffElement object graph faithfully) and *falls back to JSON* when the
   blob is unmistakably not a pickle or not valid UTF-8/JSON. Such a key is
   logged ``cross-protocol`` and left untouched; a foreign writer owns the
   bytes and we never rewrite them.

2. **Root key vs child prefix, index vs ad-hoc scan.** Keys of one tree::

       <label>:idx              SET     - every node key (maintained index)
       <label>:root:<tree_id>   STRING  - preorder rebuild order, JSON list
       <label>:node:<node_id>   STRING  - serialized node payload

   A rebuild reads ``root`` directly (O(1) order) and walks the ``idx`` set;
   ``:root:``/``:idx`` sit outside the ``:node:`` prefix so a child prefix
   scan can never return them. If ``root``/``idx`` are missing or stale, the
   *reader* pays a one-off ``:node:*`` SCAN, repairs both structures and
   continues - writers never block on index maintenance.

3. **Concurrent parent+child miss.** Fetching a missing node goes through the
   unified ``store.get_or_load_raw(key, loader)``; dedup therefore happens at
   the layer each backend actually controls (in-process event for LocalStore,
   server ``SET NX`` for RedisStore), while the observable contract is
   identical: first writer wins, duplicate writes never overwrite. An empty
   tree writes neither root nor nodes, so the first read of a genuinely empty
   tree reports ``root-empty`` instead of inventing a root key.
"""

from __future__ import annotations

import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

from diffsync.diff import Diff, DiffElement

PROTOCOL_PICKLE = "pickle"
PROTOCOL_JSON = "json"

# Structural key suffixes (tradeoff #2): node keys are the only keys under
# the ``:node:`` segment, so child prefix scans cannot collide with root/index.
SUFFIX_INDEX = "idx"
SUFFIX_ROOT = "root"
SUFFIX_NODE = "node"


def _keys(label: str, tree_id: str) -> Dict[str, str]:
    return {
        "index": f"{label}:{SUFFIX_INDEX}",
        "root": f"{label}:{SUFFIX_ROOT}:{tree_id}",
        "node_prefix": f"{label}:{SUFFIX_NODE}:",
        "node_glob": f"{label}:{SUFFIX_NODE}:*",
    }


def _node_id(element: DiffElement, depth: int, position: int) -> str:
    # Depth 0 = top-level DiffElement, depth 1 = its child; position keeps
    # same-named elements of different groups distinguishable.
    safe_name = element.name.replace("/", "_").replace(":", "_")
    return f"d{depth}-p{position:02d}-{element.type}__{safe_name}"


def _dump_element(element: DiffElement) -> Dict[str, Any]:
    """Reduce one DiffElement to JSON-safe data."""
    return {
        "type": element.type,
        "name": element.name,
        "keys": element.keys,
        "source_name": element.source_name,
        "dest_name": element.dest_name,
        "source_attrs": element.source_attrs,
        "dest_attrs": element.dest_attrs,
    }


def _build_element(data: Dict[str, Any]) -> DiffElement:
    element = DiffElement(
        obj_type=data["type"],
        name=data["name"],
        keys=dict(data["keys"]),
        source_name=data.get("source_name", "source"),
        dest_name=data.get("dest_name", "dest"),
    )
    element.add_attrs(source=data.get("source_attrs"), dest=data.get("dest_attrs"))
    return element


def encode_blob(payload: Any, protocol: str) -> bytes:
    """Serialize a node payload (or root list) with the requested protocol."""
    if protocol == PROTOCOL_JSON:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def decode_blob(blob: bytes) -> Tuple[Any, str]:
    """Decode a blob, pickle-first with JSON fallback (tradeoff #1).

    Returns:
        ``(payload, protocol_used)``; ``json`` means the JSON fallback decoded
        a key a JSON-only peer wrote.

    Raises:
        ValueError: if the blob is decodable by neither protocol.
    """
    # Fast structural sniff: protocol-2+ pickles start with the 0x80 opcode;
    # JSON text starts with an ASCII token. The exception handling below is
    # authoritative; this just avoids an exception on the common JSON path.
    if blob[:1] != b"\x80":
        try:
            return json.loads(blob.decode("utf-8")), PROTOCOL_JSON
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        return pickle.loads(blob), PROTOCOL_PICKLE  # noqa: S301
    except (pickle.UnpicklingError, UnicodeDecodeError, json.JSONDecodeError, EOFError):
        # Not a pickle we understand: retry JSON unconditionally so a JSON peer
        # key is recovered even if its first byte happened to look pickle-ish.
        try:
            return json.loads(blob.decode("utf-8")), PROTOCOL_JSON
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("blob is neither a valid pickle nor JSON payload") from exc


class DiffTreeStore:
    """Persist and rebuild a three-level Diff tree through any BaseStore."""

    def __init__(self, store: Any, *, label: str = "diffsync:diff", tree_id: str = "tree-1") -> None:
        """Bind the cache to one store backend and one logical tree id."""
        self.store = store
        self.label = label
        self.tree_id = tree_id
        self.keys = _keys(label, tree_id)

    def dump(self, diff: Diff, *, protocol: str = PROTOCOL_PICKLE) -> List[str]:
        """Write the tree: index SET, root order key, then node payloads.

        Returns:
            The preorder list of node keys written, identical to the rebuild
            order stored in the root key.
        """
        # Strict preorder: each parent key immediately precedes its subtree,
        # so rebuild attaches every child to an already-built parent in one
        # forward pass. The demo tree has three DiffElement levels
        # (site -> device -> interface); recursion is depth-agnostic.
        plan: List[Tuple[str, Dict[str, Any]]] = []
        position = 0
        prefix = self.keys["node_prefix"]

        def plan_node(element: DiffElement, depth: int) -> None:
            # Assign this node's preorder number, record it (parent before
            # children), then descend - so the set sequence IS the preorder.
            nonlocal position
            node_key = f"{prefix}{_node_id(element, depth, position)}"
            position += 1
            child_keys: List[str] = []
            # Reserve child numbers in order before recursing; each recursive
            # call consumes exactly one position, keeping ids stable.
            children = list(element.get_children())
            for offset, child in enumerate(children, start=1):
                child_keys.append(f"{prefix}{_node_id(child, depth + 1, position + offset - 1)}")
            payload = _dump_element(element)
            payload["children"] = [key.split(prefix, 1)[1] for key in child_keys]
            plan.append((node_key, payload))
            for child in children:
                plan_node(child, depth + 1)

        for top in diff.get_children():
            plan_node(top, 0)

        order = [node_key for node_key, _ in plan]
        for node_key, payload in plan:
            self.store.set_raw(node_key, encode_blob(payload, protocol))
            self.store.add_index_raw(self.keys["index"], node_key)

        if order:
            # Root is written LAST, acting as the commit marker: a reader that
            # cannot see the root yet falls back to a node scan rather than
            # observing a half-written tree. Empty trees get no root key at
            # all, so a later read reports "root empty" instead of inventing
            # a phantom tree.
            self.store.set_raw(self.keys["root"], encode_blob(order, PROTOCOL_JSON))
        return order

    def _node_order(self) -> Tuple[Optional[List[str]], str]:
        """Resolve ordered node keys, repairing root/index when stale.

        Returns:
            ``(order_or_None, source)`` with source in
            ``root-index`` / ``repaired-scan`` / ``root-empty``.
        """
        root_blob = self.store.get_raw(self.keys["root"])
        if root_blob is None:
            scanned = self.store.scan_keys_raw(self.keys["node_glob"])
            if not scanned:
                # No root and no nodes: genuinely empty tree. No back-to-source
                # scan debt is paid beyond the empty existence check.
                self.store._trace(  # noqa: SLF001
                    op="index", key=self.keys["root"], result="index-hit",
                    detail="root empty: no root key and no node keys",
                )
                return None, "root-empty"
            # Root lost but nodes survived: the READER pays the fallback scan
            # and repairs index + root; writers never block on this (tradeoff 2).
            order = sorted(scanned)
            self._repair(order)
            return order, "repaired-scan"

        order, _ = decode_blob(root_blob)
        indexed = self.store.index_members_raw(self.keys["index"])
        if indexed is not None and set(indexed) == set(order):
            self.store._trace(  # noqa: SLF001
                op="index", key=self.keys["index"], result="index-hit",
                detail=f"{len(order)} node key(s), root and index agree",
            )
            return order, "root-index"

        # Index missing/stale relative to a valid root: reconcile against a
        # fallback scan; keep the root's preorder where keys still exist.
        scanned = self.store.scan_keys_raw(self.keys["node_glob"])
        scanned_set = set(scanned)
        reconciled = [key for key in order if key in scanned_set]
        missing = sorted(scanned_set - set(order))
        order = reconciled + missing
        self._repair(order)
        return order, "repaired-scan"

    def _repair(self, order: List[str]) -> None:
        """Reader-side repair: rebuild the index SET and persist root order."""
        for key in order:
            self.store.add_index_raw(self.keys["index"], key)
        self.store.set_raw(self.keys["root"], encode_blob(order, PROTOCOL_JSON))
        self.store._trace(  # noqa: SLF001
            op="index", key=self.keys["index"], result="index-repaired",
            detail="reader paid fallback node scan and rebuilt index/root",
        )

    def load(
        self,
        *,
        protocol: str = PROTOCOL_PICKLE,
        loader: Optional[Any] = None,
    ) -> Tuple[Diff, str, List[Tuple[str, str, str]]]:
        """Rebuild the Diff tree in the stored preorder.

        Args:
            protocol: Native protocol this caller expects; a key written by
                the other protocol is still decoded (pickle-first, JSON
                fallback) and traced as ``cross-protocol`` (tradeoff #1).
            loader: Optional ``callable(node_key) -> bytes`` used as the source
                of truth on cache miss; without it a miss raises ``KeyError``.

        Returns:
            ``(diff, order_source, rebuild_log)``; ``order_source`` is one of
            ``root-index`` / ``repaired-scan`` / ``root-empty``; ``rebuild_log``
            lists ``(node_key, parent|child, outcome)`` in fetch/attach order.
        """
        order, order_source = self._node_order()
        diff = Diff()
        rebuild_log: List[Tuple[str, str, str]] = []
        if order is None:
            return diff, order_source, rebuild_log

        # Each frame is (parent element, direct children still expected). In
        # preorder the next node is consumed by the current innermost open
        # parent; when its counter reaches zero the frame closes.
        stack: List[Tuple[DiffElement, int]] = []
        for key in order:
            # Account for this node as a child of its (already open) parent.
            if stack:
                parent, remaining = stack[-1]
                stack[-1] = (parent, remaining - 1)

            if loader is None:
                blob = self.store.get_raw(key)
                if blob is None:
                    raise KeyError(f"cache miss for {key} and no loader was provided")
                loaded = False
            else:
                blob, loaded = self.store.get_or_load_raw(key, lambda bound=key: loader(bound))

            payload, used_protocol = decode_blob(blob)
            child_count = len(payload.get("children", []))

            if loaded:
                outcome = "miss"
            elif used_protocol != protocol:
                outcome = "cross-protocol"
                # Foreign-protocol blob: decoded via fallback but never
                # rewritten, so the other protocol's writer keeps ownership.
                self.store._trace(  # noqa: SLF001
                    op="get", key=key, result="cross-protocol", protocol=used_protocol,
                    detail="foreign-protocol key decoded via fallback, left untouched",
                )
            else:
                outcome = "hit"

            element = _build_element(payload)
            depth = len(stack)
            if depth == 0:
                diff.add(element)
            else:
                parent, _ = stack[-1]
                parent.add_child(element)
            if child_count:
                stack.append((element, child_count))
            # Close any parent frames whose children have all been consumed.
            while stack and stack[-1][1] == 0:
                stack.pop()
            depth_label = f"level-{depth}"
            rebuild_log.append((key, depth_label, outcome))

        return diff, order_source, rebuild_log
