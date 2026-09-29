"""Minimal in-memory Redis double used for tests and trace generation.

Only the commands exercised by :class:`~diffsync.store.redis.RedisStore` and
the raw Diff tree cache are implemented. Semantics that matter for the three
tradeoffs in :mod:`diffsync.store.tracing` are deliberately exact:

* :meth:`set` honours ``nx=True`` as the server-side dedup primitive
  (first writer wins; see tradeoff #3);
* :meth:`scan_iter` implements the ``*`` glob prefix/suffix matching used by
  the child ``:node:`` scans (tradeoff #2);
* SETs (:meth:`sadd`/:meth:`smembers`) are distinct from string keys, matching
  the index storage model.
"""

from __future__ import annotations

import fnmatch
import threading
from typing import Any, Dict, Iterable, List, Optional, Set


class FakeRedis:
    """Thread-safe, process-local stand-in for a subset of the Redis API."""

    def __init__(self) -> None:
        """Initialize empty string and set stores."""
        self._strings: Dict[str, bytes] = {}
        self._sets: Dict[str, Set[bytes]] = {}
        self._lock = threading.Lock()
        # Minimal stand-in for ``redis.Redis.connection_pool`` so helpers that
        # extract a connection path (legacy tests) keep a stable interface.
        self.connection_pool = type("FakeConnectionPool", (), {
            "connection_kwargs": {"path": "fakeredis-in-memory"}  # noqa: S108
        })()

    def ping(self) -> bool:
        """Always reachable."""
        return True

    @staticmethod
    def _as_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")

    @staticmethod
    def _as_str(key: Any) -> str:
        return key.decode("utf-8") if isinstance(key, bytes) else str(key)

    def get(self, key: str) -> Optional[bytes]:
        """String GET; returns ``None`` for missing or wrong-type keys."""
        with self._lock:
            return self._strings.get(self._as_str(key))

    def set(self, key: str, value: Any, nx: bool = False, **_: Any) -> bool:
        """String SET. With ``nx=True`` only write an absent key.

        Returns:
            True when the value was written, False on an NX conflict.
        """
        data = self._as_bytes(value)
        with self._lock:
            key = self._as_str(key)
            if nx and key in self._strings:
                return False
            self._strings[key] = data
            return True

    def exists(self, key: str) -> int:
        """Return 1 if a string or set key exists, else 0."""
        with self._lock:
            key = self._as_str(key)
            return int(key in self._strings or key in self._sets)

    def delete(self, key: str) -> int:
        """Delete a string or set key; return how many keys were removed."""
        with self._lock:
            key = self._as_str(key)
            removed = 0
            if key in self._strings:
                del self._strings[key]
                removed += 1
            if key in self._sets:
                del self._sets[key]
                removed += 1
            return removed

    def scan_iter(self, pattern: str) -> Iterable[bytes]:
        """Yield string keys matching ``pattern`` (``fnmatch`` ``*`` globs)."""
        with self._lock:
            keys = [key.encode("utf-8") for key in self._strings if fnmatch.fnmatchcase(key, pattern)]
        yield from keys

    def sadd(self, key: str, *members: Any) -> int:
        """Add members to the SET at ``key``; return newly added count."""
        with self._lock:
            key = self._as_str(key)
            bucket = self._sets.setdefault(key, set())
            added = 0
            for member in members:
                data = self._as_bytes(member)
                if data not in bucket:
                    bucket.add(data)
                    added += 1
            return added

    def smembers(self, key: str) -> Set[bytes]:
        """Return a copy of the SET at ``key`` (empty set if absent)."""
        with self._lock:
            return set(self._sets.get(self._as_str(key), set()))

    def flushall(self) -> bool:
        """Remove every key (test helper)."""
        with self._lock:
            self._strings.clear()
            self._sets.clear()
        return True

    def keys_for_trace(self, pattern: str = "*") -> List[str]:
        """Debug helper listing matching string keys in sorted order."""
        with self._lock:
            return sorted(key for key in self._strings if fnmatch.fnmatchcase(key, pattern))
