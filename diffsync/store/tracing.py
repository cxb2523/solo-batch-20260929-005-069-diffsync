"""Observability layer shared by the store backends and the Diff tree cache.

Three design tradeoffs are exercised by this module. They are deliberately kept
*mutually consistent* and are referenced from the comments of the store backends
(``diffsync/store/__init__.py``, ``diffsync/store/local.py``,
``diffsync/store/redis.py``) as well as from the generated ``trace.html`` page:

1. **pickle vs JSON dual protocol (deserialization order).**
   Both backends may hold blobs written by a peer running the other serialization
   protocol. We therefore always *try pickle first* on read (pickle preserves the
   Diff/DiffElement object graph natively) and *fall back to JSON* when the blob
   clearly is not a pickle (``pickle.UnpicklingError``) or is not UTF-8/JSON
   (``UnicodeDecodeError`` / ``json.JSONDecodeError``). A blob decoded through the
   fallback is reported as a ``cross-protocol`` hit; it is never silently rewritten
   in the native protocol, so we cannot clobber a key another writer owns.

2. **Root key naming vs parent/child prefix scanning (index vs ad-hoc scan).**
   Key layout is::

       diffsync:<store_id>:idx                # set: node keys of one tree
       diffsync:<store_id>:root:<tree_id>     # str : preorder-ordered node keys
       diffsync:<store_id>:node:<node_id>     # str : serialized node payload

   Prefix-scanning ``node:*`` is what an *index-less* rebuild would have to do, but
   it is O(N) on every rebuild. We instead *maintain an index* (``idx`` set) and a
   ``root`` key, both deliberately named outside the ``node:`` prefix so they can
   never be returned by a child prefix scan. If the index/root are missing or
   stale (e.g. only node keys survived), the *reader* pays the one-off fallback
   ``node:*`` scan, rebuilds the index, and writes it back; writers never block on
   index repair. The fallback scan explicitly strips any non-``node:`` key it
   happens to see (defense in depth for the root-vs-prefix collision).

3. **Concurrent parent+child miss: who dedups, and do duplicate writes overwrite?**
   The store *signatures* are unified, but the two backends cannot dedup at the
   same layer:

   * LocalStore shares one process, so it owns a process-local in-flight registry
     and lets concurrent loaders of the same key *share* one source fetch.
   * RedisStore cannot see another process' in-flight registry, so it pushes the
   *   guarantee down to the server with ``SET NX``: duplicate work may happen, but
   *   the first writer wins and later writes never overwrite.

   Both backends therefore converge on the same observable contract:
   **first writer wins, duplicate ``set`` on an existing key is a no-op overwrite
   (``write-skipped``)**, so a late-loaded parent cannot replace a fresher child
   subtree that landed first.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """One observed store operation.

    Attributes:
        seq: Monotonic sequence number across all stores attached to one recorder.
        store: Name of the store backend, as ``str(store)``.
        op: One of ``get`` / ``set`` / ``scan`` / ``index`` / ``rebuild``.
        key: Store key the operation targeted (may be a scan pattern for ``scan``).
        result: Outcome category; see the ``RESULT_*`` constants.
        protocol: ``pickle`` / ``json`` / ``-`` for non-payload events.
        detail: Free-form human-readable extra context.
    """

    seq: int
    store: str
    op: str
    key: str
    result: str
    protocol: str = "-"
    detail: str = ""


#: ``get`` found a value serialized in the backend's native protocol.
RESULT_HIT = "hit"
#: ``get`` found a value written by the *other* protocol and decoded it via fallback.
RESULT_CROSS = "cross-protocol"
#: ``get`` missed and the caller had to go back to the source of truth.
RESULT_MISS = "miss"
#: ``set`` wrote a previously absent key.
RESULT_WRITTEN = "written"
#: ``set`` on an existing key was skipped (first-writer-wins, see tradeoff #3).
RESULT_SKIPPED = "write-skipped"
#: A prefix scan / index operation (scan pattern or index key in ``key``).
RESULT_SCAN = "scan"
#: Index/root was found to be valid.
RESULT_INDEX_OK = "index-hit"
#: Index/root was missing/stale; reader paid the fallback scan and repaired it.
RESULT_INDEX_REPAIRED = "index-repaired"


@dataclass
class Tracer:
    """Thread-safe recorder of store operations.

    A single tracer is attached to every store involved in one scenario so that
    the global ``seq`` numbers show the true interleaving of ``get``/``set`` calls
    across LocalStore and the in-memory fake Redis.
    """

    events: List[TraceEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        store: str,
        op: str,
        key: str,
        result: str,
        protocol: str = "-",
        detail: str = "",
    ) -> TraceEvent:
        """Append one event and return it."""
        with self._lock:
            event = TraceEvent(
                seq=len(self.events) + 1,
                store=store,
                op=op,
                key=key,
                result=result,
                protocol=protocol,
                detail=detail,
            )
            self.events.append(event)
        return event

    def events_for(self, store: str) -> List[TraceEvent]:
        """Return a copy of the events recorded for one store, in order."""
        return [event for event in self.events if event.store == store]

    def hit_rate(self, store: Optional[str] = None) -> Dict[str, Any]:
        """Compute get/set statistics.

        Args:
            store: Restrict the numbers to one store name; ``None`` aggregates all.

        Returns:
            Dict with the raw counters and a ``hit_rate`` percentage (0.0-100.0).
        """
        events = self.events if store is None else self.events_for(store)
        gets = [event for event in events if event.op == "get"]
        hits = sum(1 for event in gets if event.result in (RESULT_HIT, RESULT_CROSS))
        misses = sum(1 for event in gets if event.result == RESULT_MISS)
        sets = [event for event in events if event.op == "set"]
        written = sum(1 for event in sets if event.result == RESULT_WRITTEN)
        skipped = sum(1 for event in sets if event.result == RESULT_SKIPPED)
        cross = sum(1 for event in gets if event.result == RESULT_CROSS)
        total = len(gets)
        return {
            "gets": total,
            "hits": hits,
            "cross": cross,
            "misses": misses,
            "sets": len(sets),
            "written": written,
            "skipped": skipped,
            "hit_rate": round(100.0 * hits / total, 1) if total else 0.0,
        }
