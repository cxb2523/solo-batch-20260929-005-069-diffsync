"""Generate an offline HTML trace for LocalStore and in-memory RedisStore."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from typing import Any

from diffsync.diff import Diff, DiffElement
from diffsync.store import DiffProtocol, encode_diff_node_id
from diffsync.store.local import LocalStore
from diffsync.store.redis import RedisStore


class FakeRedis:
    """Small thread-safe Redis double used without launching a Redis process."""

    def __init__(self) -> None:
        """Initialize the in-memory mapping."""
        self.data: dict[bytes, bytes] = {}
        self.lock = threading.RLock()

    def ping(self) -> bool:
        """Match Redis connectivity checks."""
        return True

    def get(self, key: Any) -> bytes | None:
        """Get one bytes value."""
        with self.lock:
            return self.data.get(self._key(key))

    def set(self, key: Any, value: Any, nx: bool = False, ex: Any = None) -> bool:
        """Set one bytes value, optionally only when the key is new."""
        del ex
        redis_key = self._key(key)
        with self.lock:
            if nx and redis_key in self.data:
                return False
            self.data[redis_key] = self._value(value)
            return True

    def exists(self, key: Any) -> int:
        """Report whether one key exists."""
        with self.lock:
            return int(self._key(key) in self.data)

    def delete(self, key: Any) -> int:
        """Delete one key."""
        with self.lock:
            return int(self.data.pop(self._key(key), None) is not None)

    def scan_iter(self, pattern: Any) -> list[bytes]:
        """List bytes keys matching a simple prefix glob."""
        prefix = pattern[:-1] if isinstance(pattern, str) and pattern.endswith("*") else pattern
        prefix_bytes = prefix.encode("utf-8") if isinstance(prefix, str) else prefix
        with self.lock:
            return [key for key in self.data if key.startswith(prefix_bytes)]

    @staticmethod
    def _key(key: Any) -> bytes:
        return key.encode("utf-8") if isinstance(key, str) else bytes(key)

    @staticmethod
    def _value(value: Any) -> bytes:
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)


def make_diff_tree() -> Diff:
    """Create a deterministic three-level Diff tree."""
    diff = Diff()
    region = DiffElement("region", "apac", {"name": "apac"}, source_name="local", dest_name="redis")
    region.add_attrs(source={"owner": "net"}, dest={"owner": "sec"})
    site = DiffElement("site", "sg", {"region": "apac", "name": "sg"}, source_name="local", dest_name="redis")
    site.add_attrs(source={"status": "active"}, dest={"status": "maintenance"})
    device = DiffElement(
        "device",
        "leaf-01",
        {"site": "sg", "name": "leaf-01"},
        source_name="local",
        dest_name="redis",
    )
    device.add_attrs(source={"role": "leaf"}, dest={"role": "spine"})
    site.add_child(device)
    region.add_child(site)
    diff.add(region)
    return diff


def dump_and_rebuild(store: Any, *, protocol: DiffProtocol) -> tuple[Diff, list[str]]:
    """Dump the tree, rebuild it, and return rebuild order."""
    source = make_diff_tree()
    store.trace_clear()
    store.dump_diff(source, tree="three-level", protocol=protocol)
    rebuilt = store.load_diff(tree="three-level")
    return rebuilt, list(store.last_diff_rebuild_order)


def fresh_element(level: str) -> DiffElement:
    """Build one deterministic source element for miss tests."""
    element = DiffElement(level, f"{level}-source", {"level": level}, source_name="source", dest_name="store")
    element.add_attrs(source={"value": level}, dest=None)
    return element


def concurrent_parent_child_miss(store: Any, *, protocol: DiffProtocol = "json") -> dict[str, Any]:
    """Force a parent and child miss at the same time and observe same-key dedupe."""
    store.trace_clear()
    parent_id = encode_diff_node_id("parent", "p1")
    child_id = f"{parent_id}/{encode_diff_node_id('child', 'c1')}"
    calls: dict[str, int] = {"parent": 0, "child": 0}
    start = threading.Barrier(4)

    def request(level: str, node_id: str) -> None:
        start.wait()
        store.get_or_set_diff_node(
            tree="misses",
            node_id=node_id,
            factory=lambda: source(level),
            protocol=protocol,
        )

    def source(level: str) -> DiffElement:
        calls[level] += 1
        return fresh_element(level)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(request, "parent", parent_id),
            executor.submit(request, "parent", parent_id),
            executor.submit(request, "child", child_id),
            executor.submit(request, "child", child_id),
        ]
        for future in futures:
            future.result()

    return {"calls": calls, "events": list(store.trace_events)}


def cross_protocol_round_trip() -> list[dict[str, Any]]:
    """Read pickle-written diff nodes through a JSON-first store and vice versa."""
    rows = []
    protocol_pairs: list[tuple[DiffProtocol, DiffProtocol]] = [("json", "pickle"), ("pickle", "json")]
    for writer_protocol, reader_protocol in protocol_pairs:
        store = LocalStore(name=f"cross-{writer_protocol}-{reader_protocol}")
        node_id = encode_diff_node_id("cross", writer_protocol)
        source = fresh_element(writer_protocol)
        store.set_diff_node(tree="cross", node_id=node_id, element=source, protocol=writer_protocol)
        read, observed_protocol, _, _ = store.get_or_set_diff_node(
            tree="cross",
            node_id=node_id,
            factory=lambda: fresh_element("should-not-run"),
            protocol=reader_protocol,
        )
        rows.append(
            {
                "written": writer_protocol,
                "requested": reader_protocol,
                "observed": observed_protocol,
                "name": read.name,
                "events": list(store.trace_events),
            }
        )
    return rows


def render_html(results: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> str:
    """Render the static trace report."""
    style = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a;background:#f8fafc}
    h1,h2{margin:0 0 12px}.card{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:16px;margin:16px 0}
    table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}
    th{background:#eef2f6}.hit{color:#116329;font-weight:700}.miss,.backfill{color:#9a3412;font-weight:700}
    .set{color:#0969da;font-weight:700}.duplicate{color:#8250df;font-weight:700}.muted{color:#57606a}
    code{background:#f6f8fa;padding:1px 4px;border-radius:4px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    .metric{background:#f1f8ff;border:1px solid #c8e1ff;border-radius:8px;padding:10px}.metric b{font-size:20px;display:block}
    """
    decisions = [
        (
            "Protocol",
            "Reads try JSON first, then pickle only when JSON decoding fails. JSON and pickle keys may coexist; "
            "the envelope identifies the actual protocol, so a reader downgrades to the writer's format instead of overwriting it.",
        ),
        (
            "Root/index/prefix",
            "Root and index are siblings of an explicit :node: prefix, avoiding root-key false positives during child scans. "
            "Normal reads use the maintained index; if it is lost, the first reader pays one scan and rewrites it.",
        ),
        (
            "Concurrent misses",
            "LocalStore deduplicates the same key in-process; RedisStore owns distributed same-key locking. Parent and child "
            "are different keys and may both call their source, while duplicate same-key writes never overwrite the winner.",
        ),
    ]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>DiffSync Store Trace</title>",
        f"<style>{style}</style></head><body><h1>DiffSync Store Trace</h1>",
        "<p class='muted'>Three-layer Diff tree, key sequence, hits, and source backfills.</p>",
    ]
    for title, body in decisions:
        parts.append(f"<div class='card'><h2>{escape(title)}</h2><p>{escape(body)}</p></div>")

    for result in results:
        summary = result["summary"]
        parts.append(f"<div class='card'><h2>{escape(result['name'])}</h2><div class='grid'>")
        for label, key in (("GETs", "gets"), ("Hits", "hits"), ("Misses/backfills", "misses"), ("Hit rate", "hit_rate")):
            value = f"{summary[key]:.1%}" if key == "hit_rate" else summary[key]
            parts.append(f"<div class='metric'><span>{label}</span><b>{value}</b></div>")
        parts.append("</div>")
        parts.append("<h3>Rebuild order</h3><p>" + " → ".join(f"<code>{escape(item)}</code>" for item in result["order"]) + "</p>")
        parts.append("<h3>Key sequence</h3><table><thead><tr><th>#</th><th>Node</th><th>Operation</th><th>Key</th><th>Result</th><th>Protocol</th></tr></thead><tbody>")
        for event in result["events"]:
            parts.append(
                "<tr>"
                f"<td>{event.sequence}</td><td><code>{escape(event.node or '')}</code></td>"
                f"<td>{event.operation}</td><td><code>{escape(event.key)}</code></td>"
                f"<td class='{event.result}'>{event.result}</td><td>{event.protocol or ''}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    for result in results:
        calls = result["calls"]
        parts.append(
            f"<div class='card'><h2>{escape(result['name'])} parent/child miss</h2>"
            f"<p>Source calls: parent={calls['parent']}, child={calls['child']}; one source call per distinct node.</p>"
            "<table><thead><tr><th>#</th><th>Node</th><th>Operation</th><th>Key</th><th>Result</th><th>Protocol</th></tr></thead><tbody>"
        )
        for event in result["miss_events"]:
            parts.append(
                "<tr>"
                f"<td>{event.sequence}</td><td><code>{escape(event.node or '')}</code></td>"
                f"<td>{event.operation}</td><td><code>{escape(event.key)}</code></td>"
                f"<td class='{event.result}'>{event.result}</td><td>{event.protocol or ''}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    parts.append("<div class='card'><h2>Cross-protocol reads</h2><table><thead><tr><th>Written</th><th>Requested</th><th>Observed</th><th>Node</th><th>Result</th></tr></thead><tbody>")
    for row in cross_rows:
        event = next(event for event in row["events"] if event.operation == "get" and event.result == "hit")
        parts.append(
            f"<tr><td>{row['written']}</td><td>{row['requested']}</td><td>{row['observed']}</td>"
            f"<td>{escape(row['name'])}</td><td class='hit'>{event.result}</td></tr>"
        )
    parts.append("</tbody></table></div></body></html>")
    return "".join(parts)


def main() -> None:
    """Generate trace.html in the current working directory."""
    fake_redis = FakeRedis()
    stores: list[tuple[Any, DiffProtocol]] = [
        (LocalStore(name="LocalStore (JSON)"), "json"),
        (RedisStore(name="Memory RedisStore (pickle)", store_id="trace", client=fake_redis), "pickle"),
    ]
    results = []
    for store, protocol in stores:
        rebuilt, order = dump_and_rebuild(store, protocol=protocol)
        if rebuilt.groups() != ["region"] or len(rebuilt) != 3:
            raise RuntimeError("three-level diff tree did not rebuild correctly")
        dump_events = list(store.trace_events)
        miss = concurrent_parent_child_miss(store, protocol=protocol)
        gets = [event for event in dump_events if event.operation == "get"]
        hits = [event for event in gets if event.result == "hit"]
        results.append(
            {
                "name": str(store),
                "events": dump_events,
                "summary": {
                    "gets": len(gets),
                    "hits": len(hits),
                    "misses": len(gets) - len(hits),
                    "hit_rate": len(hits) / len(gets),
                },
                "order": order,
                "calls": miss["calls"],
                "miss_events": miss["events"],
            }
        )

    output = Path("trace.html")
    output.write_text(render_html(results, cross_protocol_round_trip()), encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
