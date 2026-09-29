#!/usr/bin/env python
"""Generate trace.html: observability of the Diff tree cache over both stores.

Run with ``python trace_store.py``. It builds a three-level Diff tree
(site -> device -> interface), dumps it, and drives it through:

* :class:`~diffsync.store.local.LocalStore`;
* :class:`~diffsync.store.redis.RedisStore` backed by the in-memory
  :class:`~diffsync.store.fake_redis.FakeRedis` (no Redis server required).

The page lists, per node, every get/set key in sequence, hit vs back-to-source
classification, and the whole-tree rebuild order, together with hit rates and
the three mutually-binding tradeoffs (also documented in
``diffsync/store/tracing.py`` and the store backends).
"""

# ruff: noqa: S101 - executable scenario/demo script; asserts guard demo invariants.

from __future__ import annotations

import html
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from diffsync.diff import Diff, DiffElement
from diffsync.store.diff_tree import (
    PROTOCOL_JSON,
    PROTOCOL_PICKLE,
    DiffTreeStore,
    decode_blob,
    encode_blob,
)
from diffsync.store.fake_redis import FakeRedis
from diffsync.store.local import LocalStore
from diffsync.store.redis import RedisStore
from diffsync.store.tracing import Tracer

TREE_ID = "demo"
LABEL = "diffsync:diff"


def build_tree() -> Diff:
    """Construct the three-level Diff/DiffElement tree used in every scenario."""
    diff = Diff()

    site = DiffElement("site", "nyc", {"name": "nyc"})
    site.add_attrs(source={"city": "New York"}, dest={"city": "NYC"})

    device = DiffElement("device", "spine1", {"name": "spine1"})
    device.add_attrs(source={"role": "spine", "vendor": "acme"}, dest={"role": "leaf"})

    interface = DiffElement(
        "interface", "eth0", {"device_name": "spine1", "name": "eth0"}
    )
    interface.add_attrs(source={"vlan": 10, "mtu": 9000}, dest={"vlan": 20, "mtu": 1500})

    device.add_child(interface)
    site.add_child(device)
    diff.add(site)
    return diff


@dataclass
class ScenarioResult:
    """Everything the HTML page renders for one store x scenario run."""

    store_name: str
    scenario: str
    description: str
    events: List[Any]
    rebuild_order: List[Tuple[str, str, str]] = field(default_factory=list)
    order_source: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    loader_calls: int = 0
    loader_keys: List[str] = field(default_factory=list)
    note: str = ""


def _snapshot_source(store: Any, cache: DiffTreeStore, diff: Diff, protocol: str) -> Dict[str, bytes]:
    """Dump the tree once and return the immutable source-of-truth blobs."""
    order = cache.dump(diff, protocol=protocol)
    source: Dict[str, bytes] = {}
    for key in order:
        source[key] = store.get_raw(key)
    return source


def scenario_warm(store: Any, diff: Diff, protocol: str) -> ScenarioResult:
    """Dump then rebuild from a warm cache: all node gets hit."""
    cache = DiffTreeStore(store, tree_id=TREE_ID)
    cache.dump(diff, protocol=protocol)
    rebuilt, source, order = cache.load(protocol=protocol)
    assert len(rebuilt) and len(order) == 3
    return ScenarioResult(
        store_name=str(store),
        scenario="warm-rebuild",
        description="Dump then rebuild from a warm cache; root + index present, every node get is a hit.",
        events=list(store.tracer.events),
        rebuild_order=order,
        order_source=source,
        stats=store.tracer.hit_rate(),
        note="Key sequence = set nodes -> set root; rebuild = get root -> index -> node gets (preorder).",
    )


def scenario_empty_root(store: Any, protocol: str) -> ScenarioResult:
    """Dump nothing and rebuild: root key is absent, tree is genuinely empty."""
    cache = DiffTreeStore(store, tree_id="empty")
    rebuilt, source, order = cache.load(protocol=protocol)
    assert source == "root-empty" and not order and len(rebuilt) == 0
    return ScenarioResult(
        store_name=str(store),
        scenario="root-empty",
        description="Empty tree writes no root key; rebuild finds root absent and zero node keys, so no phantom tree is invented.",
        events=list(store.tracer.events),
        rebuild_order=order,
        order_source=source,
        stats=store.tracer.hit_rate(),
        note="Cost paid: one root GET + one empty node-prefix existence scan; no index repair needed.",
    )


def scenario_index_repair(store: Any, diff: Diff, protocol: str) -> ScenarioResult:
    """Dump, then lose root+index: reader pays the fallback scan and repairs."""
    cache = DiffTreeStore(store, tree_id="repair")
    cache.dump(diff, protocol=protocol)
    client_raw = getattr(store, "_raw", None)
    if client_raw is not None:
        client_raw.pop(cache.keys["root"], None)
        store._raw_index.pop(cache.keys["index"], None)  # noqa: SLF001
    else:
        store._store.delete(cache.keys["root"])  # noqa: SLF001
        store._store.delete(cache.keys["index"])  # noqa: SLF001

    rebuilt, source, order = cache.load(protocol=protocol)
    assert source == "repaired-scan" and len(order) == 3
    # Second read now uses the repaired index/root.
    rebuilt2, source2, order2 = cache.load(protocol=protocol)
    assert source2 == "root-index" and len(order2) == 3
    return ScenarioResult(
        store_name=str(store),
        scenario="index-repair",
        description="Nodes survive but root and index are lost; the reader does a one-off :node:* scan, rebuilds index+root, and the next rebuild is O(1) again.",
        events=list(store.tracer.events),
        rebuild_order=order,
        order_source=source,
        stats=store.tracer.hit_rate(),
        note=f"First rebuild order source: {source}; second rebuild order source: {source2} (repaired).",
    )


def scenario_concurrent_miss(store: Any, diff: Diff, protocol: str) -> ScenarioResult:
    """Parent and child node keys miss simultaneously, root/index stay intact."""
    cache = DiffTreeStore(store, tree_id="cold")
    source = _snapshot_source(store, cache, diff, protocol)
    # Keep root + index in place (so rebuild order is known) but drop EVERY
    # node payload: parent and both children miss at the same instant. This is
    # exactly the "parent+child concurrent miss" the dedup tradeoff targets.
    node_keys = list(source)
    raw = getattr(store, "_raw", None)
    if raw is not None:
        for key in node_keys:
            raw.pop(key, None)
    else:
        for key in node_keys:
            store._store.delete(key)  # noqa: SLF001

    loader_calls: List[str] = []
    loader_lock = threading.Lock()
    start_gate = threading.Event()

    def source_loader(key: str) -> bytes:
        # Hold every loader at the same gate and add latency afterwards so the
        # two rebuild threads are inside the miss handlers simultaneously.
        start_gate.wait(timeout=5)
        with loader_lock:
            loader_calls.append(key)
        threading.Event().wait(0.05)
        return source[key]

    rebuilds: List[Tuple[Diff, str, list]] = []
    errors: List[BaseException] = []

    def worker(key_order: List[str]) -> None:
        try:
            local_cache = DiffTreeStore(store, tree_id="cold")
            # Rebuild in a shuffled starting point so parent/child loads overlap.
            result = local_cache.load(protocol=protocol, loader=source_loader)
            rebuilds.append(result)
        except BaseException as exc:  # noqa: BLE001 - surfaced into the report
            errors.append(exc)

    keys = list(source)
    threads = [
        threading.Thread(target=worker, args=(keys,)),
        threading.Thread(target=worker, args=(list(reversed(keys)),)),
    ]
    for thread in threads:
        thread.start()
    start_gate.set()
    for thread in threads:
        thread.join()
    assert not errors, errors

    unique_loads = len(set(loader_calls))
    dedup_note = (
        f"LocalStore dedups same-key misses in-process: {len(loader_calls)} loader call(s) "
        f"for {unique_loads} missing node key(s); the second thread shares the in-flight result."
        if isinstance(store, LocalStore)
        else f"RedisStore cannot share an in-process event: {len(loader_calls)} loader call(s) "
        f"for {unique_loads} missing key(s); SET NX keeps 3 first writes and skips the duplicate writes, never overwriting."
    )
    return ScenarioResult(
        store_name=str(store),
        scenario="concurrent-parent-child-miss",
        description="Cold cache, two rebuild threads start simultaneously so a parent and its children miss together.",
        events=list(store.tracer.events),
        rebuild_order=rebuilds[0][2],
        order_source=rebuilds[0][1],
        stats=store.tracer.hit_rate(),
        loader_calls=len(loader_calls),
        loader_keys=loader_calls,
        note=dedup_note,
    )


def scenario_cross_protocol(
    store_factory: Callable[[str, Tracer], Any],
    store_name: str,
    diff: Diff,
) -> ScenarioResult:
    """One protocol writes the keys, the other protocol reads them back."""
    # Writer side: pickle dump.
    writer_tracer = Tracer()
    writer = store_factory(f"{store_name}-writer", writer_tracer)
    writer_cache = DiffTreeStore(writer, tree_id="cross")
    order = writer_cache.dump(diff, protocol=PROTOCOL_PICKLE)

    # Hand the raw blobs to a second, JSON-native backend instance.
    reader_tracer = Tracer()
    reader = store_factory(f"{store_name}-reader", reader_tracer)
    reader_cache = DiffTreeStore(reader, tree_id="cross")
    blobs = {key: writer.get_raw(key) for key in order + [writer_cache.keys["index"], writer_cache.keys["root"]] if writer.get_raw(key) is not None}
    # Copy index + root + node bytes verbatim (simulating a peer's writes).
    for key, blob in blobs.items():
        if key == writer_cache.keys["index"]:
            for member in writer.index_members_raw(key):
                reader.add_index_raw(reader_cache.keys["index"], member)
        else:
            reader.set_raw(key, blob)

    rebuilt, source, rebuild_log = reader_cache.load(protocol=PROTOCOL_JSON, loader=None)
    assert len(rebuild_log) == 3
    assert all(outcome == "cross-protocol" for _, _, outcome in rebuild_log), rebuild_log

    # A native-JSON re-encode of one foreign key must NOT overwrite the pickle
    # owner's bytes: writes use first-writer-wins, so the bytes stay pickle.
    foreign_key = order[0]
    wrote = reader.set_raw(foreign_key, encode_blob({"replacement": True}, PROTOCOL_JSON), nx=True)
    assert wrote is False
    decoded, used = decode_blob(reader.get_raw(foreign_key))
    assert used == PROTOCOL_PICKLE

    return ScenarioResult(
        store_name=str(reader),
        scenario="cross-protocol-read",
        description="Pickle writer, JSON reader: deserialization tries pickle first, falls back to JSON; foreign keys decode but are never overwritten.",
        events=list(reader_tracer.events),
        rebuild_order=rebuild_log,
        order_source=source,
        stats=reader_tracer.hit_rate(),
        note=f"All 3 nodes decoded as cross-protocol; a JSON re-write was write-skipped ({wrote}) so the pickle bytes survived.",
    )


TRADEOFFS = [
    (
        "1. pickle 与 JSON 双协议共存：先试谁，读到对方的键怎么降级",
        "读取时先尝试 pickle（能无损还原 Diff/DiffElement 对象图）；若 blob 不是 pickle "
        "(UnpicklingError) 或不是合法 UTF-8/JSON，则降级用 json.loads 再解一次。"
        "被降级解出的键标记为 cross-protocol，只读不重写：键的字节归写下它的那个协议所有，"
        "本端不会把它偷偷改写成自己的协议。页面上这类命中单独计数，仍计入命中率。",
    ),
    (
        "2. root 键命名 vs 父子键前缀扫描：临时遍历还是维护索引，失效后谁付代价",
        "键布局把 idx(集合) 和 root:<tree>(顺序) 放在 node: 前缀之外，子节点扫描 "
        "node:* 在命名上就不可能扫到 root/index，后端还会再过滤一层 :node:。"
        "常规重建 O(1) 读 root 拿前序、用 idx 集合校验；一旦 root/idx 丢失或与节点不一致，"
        "代价由读者承担：一次性 node:* 全量扫描并回建索引与 root，下一次读又回到 O(1)；"
        "写入方从不会为索引修复阻塞。root 键缺失且一个节点都没有时，判定为真空树 root-empty。",
    ),
    (
        "3. 签名统一后，父子键同时未命中的并发回源：谁去重、重复写要不要覆盖",
        "BaseStore 统一 get_or_load_raw(key, loader) 签名，但去重落在各自真正掌控的层："
        "LocalStore 同进程，持有 in-flight 注册表 + 锁，同键并发只回源一次，后到者共享结果；"
        "RedisStore 看不到别的进程，把保证下沉到服务端 SET NX——可能重复算，但只有首个写入生效，"
        "竞态失败者改为读出胜者的值。两家可观测契约一致：first writer wins，重复写入不覆盖"
        "(write-skipped)，迟到的父节点不可能盖掉先落地的更新子树。",
    ),
]


def _local_factory(name: str, tracer: Tracer) -> LocalStore:
    return LocalStore(name=name, tracer=tracer)


def _redis_factory(name: str, tracer: Tracer) -> RedisStore:
    return RedisStore(name=name, client=FakeRedis(), tracer=tracer)


def run_all_scenarios() -> Dict[str, List[ScenarioResult]]:
    """Execute every scenario against both backends and group by store kind."""
    diff = build_tree()
    results: Dict[str, List[ScenarioResult]] = {"LocalStore": [], "RedisStore": []}

    local_warm = LocalStore(name="LocalStore/warm", tracer=Tracer())
    results["LocalStore"].append(scenario_warm(local_warm, diff, PROTOCOL_PICKLE))

    local_empty = LocalStore(name="LocalStore/empty", tracer=Tracer())
    results["LocalStore"].append(scenario_empty_root(local_empty, PROTOCOL_PICKLE))

    local_repair = LocalStore(name="LocalStore/repair", tracer=Tracer())
    results["LocalStore"].append(scenario_index_repair(local_repair, diff, PROTOCOL_PICKLE))

    local_cold = LocalStore(name="LocalStore/cold", tracer=Tracer())
    results["LocalStore"].append(scenario_concurrent_miss(local_cold, diff, PROTOCOL_PICKLE))

    results["LocalStore"].append(scenario_cross_protocol(_local_factory, "LocalStore", diff))

    redis_warm = RedisStore(name="RedisStore/warm", client=FakeRedis(), tracer=Tracer())
    results["RedisStore"].append(scenario_warm(redis_warm, diff, PROTOCOL_JSON))

    redis_empty = RedisStore(name="RedisStore/empty", client=FakeRedis(), tracer=Tracer())
    results["RedisStore"].append(scenario_empty_root(redis_empty, PROTOCOL_JSON))

    redis_repair = RedisStore(name="RedisStore/repair", client=FakeRedis(), tracer=Tracer())
    results["RedisStore"].append(scenario_index_repair(redis_repair, diff, PROTOCOL_JSON))

    redis_cold = RedisStore(name="RedisStore/cold", client=FakeRedis(), tracer=Tracer())
    results["RedisStore"].append(scenario_concurrent_miss(redis_cold, diff, PROTOCOL_JSON))

    results["RedisStore"].append(scenario_cross_protocol(_redis_factory, "RedisStore", diff))

    return results


RESULT_CLASS = {
    "hit": "hit",
    "cross-protocol": "cross",
    "miss": "miss",
    "written": "written",
    "write-skipped": "skipped",
    "scan": "scan",
    "index-hit": "scan",
    "index-repaired": "repair",
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _render_scenario(result: ScenarioResult) -> str:
    rows = []
    for event in result.events:
        css = RESULT_CLASS.get(event.result, "")
        rows.append(
            "<tr>"
            f"<td class='num'>{event.seq}</td>"
            f"<td class='op op-{_esc(event.op)}'>{_esc(event.op)}</td>"
            f"<td class='key'>{_esc(event.key)}</td>"
            f"<td class='res {css}'>{_esc(event.result)}</td>"
            f"<td>{_esc(event.protocol)}</td>"
            f"<td class='detail'>{_esc(event.detail)}</td>"
            "</tr>"
        )
    stats = result.stats
    stat_html = (
        f"gets={stats.get('gets', 0)} · hits={stats.get('hits', 0)} "
        f"(cross={stats.get('cross', 0)}) · misses={stats.get('misses', 0)} · "
        f"sets={stats.get('sets', 0)} · written={stats.get('written', 0)} · "
        f"skipped={stats.get('skipped', 0)} · "
        f"<b>hit-rate={stats.get('hit_rate', 0.0)}%</b>"
    )
    order_rows = []
    for idx, (key, depth, outcome) in enumerate(result.rebuild_order, start=1):
        order_rows.append(
            "<tr>"
            f"<td class='num'>{idx}</td>"
            f"<td>{_esc(depth)}</td>"
            f"<td class='key'>{_esc(key)}</td>"
            f"<td class='res {RESULT_CLASS.get(outcome, '')}'>{_esc(outcome)}</td>"
            "</tr>"
        )
    loader_html = ""
    if result.scenario == "concurrent-parent-child-miss":
        loader_html = (
            f"<p class='note'>loader 回源调用次数：<b>{result.loader_calls}</b> "
            f"(节点数 3；调用序列: {_esc(' -> '.join(k.split(':node:')[-1] for k in result.loader_keys))})</p>"
        )
    return f"""
    <section class="scenario">
      <h3>{_esc(result.scenario)} <span class="store">{_esc(result.store_name)}</span></h3>
      <p class="desc">{_esc(result.description)}</p>
      <p class="stats">{stat_html}</p>
      <p class="note">重建顺序来源：<b>{_esc(result.order_source or '-')}</b></p>
      {loader_html}
      <details open>
        <summary>每次 get/set/scan 的键序列（{len(result.events)} 条）</summary>
        <table class="events">
          <thead><tr><th>#</th><th>op</th><th>key</th><th>结果</th><th>协议</th><th>说明</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </details>
      <details open>
        <summary>整棵树的重建顺序（节点）</summary>
        <table class="order">
          <thead><tr><th>#</th><th>层级</th><th>node key</th><th>命中/回源</th></tr></thead>
          <tbody>{''.join(order_rows) or '<tr><td colspan=4>（空树，无节点）</td></tr>'}</tbody>
        </table>
      </details>
      <p class="note">{_esc(result.note)}</p>
    </section>
    """


def render_html(results: Dict[str, List[ScenarioResult]]) -> str:
    """Render the full, self-contained, double-click-openable HTML page."""
    columns = []
    for backend, scenario_results in results.items():
        rendered = "".join(_render_scenario(result) for result in scenario_results)
        overall_gets = sum(result.stats.get("gets", 0) for result in scenario_results)
        overall_hits = sum(result.stats.get("hits", 0) for result in scenario_results)
        overall_cross = sum(result.stats.get("cross", 0) for result in scenario_results)
        overall_rate = round(100.0 * overall_hits / overall_gets, 1) if overall_gets else 0.0
        columns.append(
            f"<div class='column'><h2>{_esc(backend)}</h2>"
            f"<p class='overall'>聚合命中率：<b>{overall_rate}%</b> "
            f"({overall_hits}/{overall_gets}，其中跨协议命中 {overall_cross})</p>{rendered}</div>"
        )

    tradeoff_html = "".join(
        f"<details class='tradeoff'><summary>{_esc(title)}</summary><p>{_esc(body)}</p></details>"
        for title, body in TRADEOFFS
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diff Store Trace - get/set 键序列与命中率</title>
<style>
  :root {{ --hit:#1a7f37; --cross:#9a6700; --miss:#cf222e; --written:#0969da;
          --skipped:#8250df; --scan:#57606a; --repair:#bc4c00; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0;
         background: #f6f8fa; color: #1f2328; }}
  header {{ background: #24292f; color: #fff; padding: 20px 28px; }}
  header h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 2px 0; color: #c9d1d9; font-size: 13px; }}
  main {{ padding: 20px 28px; }}
  .tradeoffs {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
                padding: 14px 18px; margin-bottom: 22px; }}
  .tradeoffs h2 {{ margin-top: 0; }}
  .tradeoff {{ margin: 8px 0; }}
  .tradeoff summary {{ font-weight: 600; cursor: pointer; }}
  .tradeoff p {{ margin: 6px 0 0; line-height: 1.6; color: #424a53; }}
  .columns {{ display: flex; gap: 20px; align-items: flex-start; }}
  .column {{ flex: 1; min-width: 0; }}
  h2 {{ border-bottom: 2px solid #d0d7de; padding-bottom: 6px; }}
  .overall {{ font-size: 14px; }}
  .scenario {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
               padding: 12px 16px; margin-bottom: 18px; }}
  .scenario h3 {{ margin: 4px 0; font-size: 16px; }}
  .store {{ float: right; font-size: 12px; color: #57606a; font-weight: 400; }}
  .desc {{ color: #424a53; font-size: 13px; }}
  .stats, .note {{ font-size: 13px; margin: 4px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
  th, td {{ border-bottom: 1px solid #eaeef2; padding: 4px 6px; text-align: left;
           vertical-align: top; word-break: break-all; }}
  th {{ background: #f6f8fa; position: sticky; top: 0; }}
  .num {{ width: 34px; color: #57606a; }}
  .key {{ font-family: Consolas, monospace; }}
  .detail {{ color: #57606a; }}
  .res {{ font-weight: 700; white-space: nowrap; }}
  .hit {{ color: var(--hit); }}
  .cross {{ color: var(--cross); }}
  .miss {{ color: var(--miss); }}
  .written {{ color: var(--written); }}
  .skipped {{ color: var(--skipped); }}
  .scan {{ color: var(--scan); }}
  .repair {{ color: var(--repair); }}
  .op {{ font-family: Consolas, monospace; white-space: nowrap; }}
  details summary {{ cursor: pointer; font-size: 13px; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>Store 可观测 Trace：三层 Diff/DiffElement 树 · get/set 键序列 · 命中率</h1>
  <p>同一棵 site(level-0) → device(level-1) → interface(level-2) 树，dump 后分别走 LocalStore 与内存假 Redis (FakeRedis)。</p>
  <p>覆盖：热重建、root 为空、父子同时未命中的并发回源、索引失效回源修复、pickle/JSON 双协议交叉读。</p>
</header>
<main>
  <div class="tradeoffs">
    <h2>三处取舍（互相咬住，注释与本页同源）</h2>
    {tradeoff_html}
  </div>
  <div class="columns">{''.join(columns)}</div>
</main>
</body>
</html>
"""


def main() -> None:
    """Run scenarios and write trace.html next to this script."""
    results = run_all_scenarios()
    output = render_html(results)
    with open("trace.html", "w", encoding="utf-8") as handle:
        handle.write(output)
    print("trace.html written")
    for backend, scenario_results in results.items():
        for result in scenario_results:
            stats = result.stats
            print(
                f"  {backend:11s} {result.scenario:28s} "
                f"gets={stats.get('gets', 0):2d} hits={stats.get('hits', 0):2d} "
                f"miss={stats.get('misses', 0):2d} skipped={stats.get('skipped', 0)} "
                f"rate={stats.get('hit_rate', 0.0)}%"
            )


if __name__ == "__main__":
    main()
