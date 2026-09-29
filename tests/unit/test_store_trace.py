"""Tests for the observable Diff tree cache and the unified raw store API.

Covers the three tradeoffs in :mod:`diffsync.store.tracing` on both backends
(LocalStore and RedisStore over an in-memory FakeRedis):

* empty tree (no root key) reads back as ``root-empty``;
* parent and child node keys missing together trigger concurrent back-to-source
  loads, deduped in-process by LocalStore and server-side (SET NX) by Redis;
* pickle/JSON cross-protocol writes are decoded by the opposite reader and are
  never overwritten;
* index/root loss is repaired by a reader-paid prefix scan.
"""

from __future__ import annotations

import threading

import pytest

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
from diffsync.store.tracing import (
    RESULT_CROSS,
    RESULT_HIT,
    RESULT_INDEX_REPAIRED,
    RESULT_MISS,
    Tracer,
)

LABEL = "diffsync:diff"


def _three_level_tree() -> Diff:
    """Build site(level-0) -> device(level-1) -> interface(level-2)."""
    diff = Diff()
    site = DiffElement("site", "nyc", {"name": "nyc"})
    site.add_attrs(source={"city": "New York"}, dest={"city": "NYC"})
    device = DiffElement("device", "spine1", {"name": "spine1"})
    device.add_attrs(source={"role": "spine"}, dest={"role": "leaf"})
    interface = DiffElement("interface", "eth0", {"device_name": "spine1", "name": "eth0"})
    interface.add_attrs(source={"vlan": 10}, dest={"vlan": 20})
    device.add_child(interface)
    site.add_child(device)
    diff.add(site)
    return diff


@pytest.fixture(name="store_factories")
def fixture_store_factories():
    """Return fresh LocalStore and FakeRedis-backed RedisStore constructors."""

    def make_local(name, tracer=None):
        return LocalStore(name=name, tracer=tracer or Tracer())

    def make_redis(name, tracer=None):
        return RedisStore(name=name, client=FakeRedis(), tracer=tracer or Tracer())

    return {"local": make_local, "redis": make_redis}


def _assert_tree_shape(diff: Diff) -> None:
    sites = list(diff.get_children())
    assert [site.type for site in sites] == ["site"]
    devices = list(sites[0].get_children())
    assert [device.type for device in devices] == ["device"]
    interfaces = list(devices[0].get_children())
    assert [interface.type for interface in interfaces] == ["interface"]
    assert interfaces[0].get_attrs_diffs()["+"]["vlan"] == 10


@pytest.mark.parametrize("backend", ["local", "redis"])
@pytest.mark.parametrize("protocol", [PROTOCOL_PICKLE, PROTOCOL_JSON])
def test_dump_and_warm_rebuild(store_factories, backend, protocol):
    store = store_factories[backend]("warm")
    cache = DiffTreeStore(store, tree_id="warm")
    order = cache.dump(_three_level_tree(), protocol=protocol)
    assert len(order) == 3
    assert [key for key in order if key.endswith("site__nyc")]

    rebuilt, source, rebuild_log = cache.load(protocol=protocol)
    assert source == "root-index"
    assert [depth for _, depth, _ in rebuild_log] == ["level-0", "level-1", "level-2"]
    assert all(outcome == RESULT_HIT for _, _, outcome in rebuild_log)
    _assert_tree_shape(rebuilt)
    assert store.tracer.hit_rate()["hit_rate"] == 100.0


@pytest.mark.parametrize("backend", ["local", "redis"])
def test_root_empty(store_factories, backend):
    store = store_factories[backend]("empty")
    cache = DiffTreeStore(store, tree_id="empty")
    # An empty Diff writes neither root key nor nodes.
    assert cache.dump(Diff()) == []

    rebuilt, source, rebuild_log = cache.load()
    assert source == "root-empty"
    assert rebuild_log == []
    assert len(rebuilt) == 0

    # The root GET misses; one empty node-prefix existence check is performed;
    # no index repair write happens because nothing is broken.
    tracer = store.tracer
    miss_events = [event for event in tracer.events if event.op == "get" and event.result == RESULT_MISS]
    assert any("root" in event.key for event in miss_events)
    assert not [event for event in tracer.events if event.result == RESULT_INDEX_REPAIRED]


@pytest.mark.parametrize("backend", ["local", "redis"])
def test_index_and_root_loss_repaired_by_reader(store_factories, backend):
    store = store_factories[backend]("repair")
    cache = DiffTreeStore(store, tree_id="repair")
    cache.dump(_three_level_tree(), protocol=PROTOCOL_JSON)

    # Simulate losing everything except the node payloads.
    raw = getattr(store, "_raw", None)
    if raw is not None:
        raw.pop(cache.keys["root"], None)
        store._raw_index.pop(cache.keys["index"], None)  # noqa: SLF001
    else:
        store._store.delete(cache.keys["root"])  # noqa: SLF001
        store._store.delete(cache.keys["index"])  # noqa: SLF001

    rebuilt, source, rebuild_log = cache.load(protocol=PROTOCOL_JSON)
    assert source == "repaired-scan"
    assert len(rebuild_log) == 3
    _assert_tree_shape(rebuilt)
    assert any(event.result == RESULT_INDEX_REPAIRED for event in store.tracer.events)

    # Second read uses the repaired root/index instead of scanning again.
    _, source2, _ = cache.load(protocol=PROTOCOL_JSON)
    assert source2 == "root-index"


@pytest.mark.parametrize("backend", ["local", "redis"])
def test_concurrent_parent_and_child_miss(store_factories, backend):
    store = store_factories[backend]("cold")
    cache = DiffTreeStore(store, tree_id="cold")
    source = {}
    for key in cache.dump(_three_level_tree(), protocol=PROTOCOL_PICKLE):
        source[key] = store.get_raw(key)

    # Delete the node payloads only; root and index remain, so rebuild order is
    # known while every parent/child node GET misses simultaneously.
    node_keys = list(source)
    raw = getattr(store, "_raw", None)
    if raw is not None:
        for key in node_keys:
            raw.pop(key, None)
    else:
        for key in node_keys:
            store._store.delete(key)  # noqa: SLF001

    loader_calls = []
    calls_lock = threading.Lock()
    gate = threading.Event()

    def loader(key):
        gate.wait(timeout=5)
        with calls_lock:
            loader_calls.append(key)
        threading.Event().wait(0.02)
        return source[key]

    errors = []

    def rebuild():
        try:
            DiffTreeStore(store, tree_id="cold").load(protocol=PROTOCOL_PICKLE, loader=loader)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=rebuild),
        threading.Thread(target=rebuild),
    ]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join()
    assert not errors

    rebuilt, source_type, _ = DiffTreeStore(store, tree_id="cold").load(protocol=PROTOCOL_PICKLE)
    _assert_tree_shape(rebuilt)

    stats = store.tracer.hit_rate()
    if backend == "local":
        # Same-process in-flight registry shares each source fetch: exactly
        # one loader call per missing node key, no duplicate writes.
        assert sorted(loader_calls) == sorted(node_keys)
        assert stats["skipped"] == 0
    else:
        # Redis-side SET NX: duplicate loaders may run, but the duplicate
        # writes are skipped (first writer wins, never overwritten).
        assert len(loader_calls) >= len(node_keys)
        assert stats["skipped"] >= len(node_keys)
    assert stats["misses"] >= len(node_keys)


@pytest.mark.parametrize("backend", ["local", "redis"])
def test_cross_protocol_read_pickle_writer_json_reader(store_factories, backend):
    writer = store_factories[backend]("writer")
    writer_cache = DiffTreeStore(writer, tree_id="cross")
    order = writer_cache.dump(_three_level_tree(), protocol=PROTOCOL_PICKLE)

    # A JSON-native reader sees the pickle bytes written by the peer.
    reader_tracer = Tracer()
    if backend == "local":
        reader = LocalStore(name="reader", tracer=reader_tracer)
        for key, blob in writer._raw.items():  # noqa: SLF001
            reader._raw[key] = blob  # noqa: SLF001
        reader._raw_index[writer_cache.keys["index"]] = set(  # noqa: SLF001
            writer._raw_index[writer_cache.keys["index"]]  # noqa: SLF001
        )
    else:
        reader = RedisStore(name="reader", client=writer._store, tracer=reader_tracer)  # noqa: SLF001

    reader_cache = DiffTreeStore(reader, tree_id="cross")
    rebuilt, source, rebuild_log = reader_cache.load(protocol=PROTOCOL_JSON)
    assert source == "root-index"
    assert [outcome for _, _, outcome in rebuild_log] == [RESULT_CROSS] * 3
    _assert_tree_shape(rebuilt)

    # The pickle-first/JSON-fallback rule decoded the bytes; protocol sniffing
    # reports JSON used and must not have rewritten the foreign key.
    for key in order:
        _, used = decode_blob(reader.get_raw(key))
        assert used == PROTOCOL_PICKLE

    # A JSON re-write of the foreign key is skipped (no overwrite).
    assert reader.set_raw(order[0], encode_blob({"x": 1}, PROTOCOL_JSON), nx=True) is False
    _, used = decode_blob(reader.get_raw(order[0]))
    assert used == PROTOCOL_PICKLE
    assert reader_tracer.hit_rate()["hit_rate"] == 100.0


@pytest.mark.parametrize("backend", ["local", "redis"])
def test_cross_protocol_read_json_writer_pickle_reader(store_factories, backend):
    writer = store_factories[backend]("writer-json")
    writer_cache = DiffTreeStore(writer, tree_id="cross2")
    order = writer_cache.dump(_three_level_tree(), protocol=PROTOCOL_JSON)

    reader_tracer = Tracer()
    if backend == "local":
        reader = LocalStore(name="reader-pickle", tracer=reader_tracer)
        for key, blob in writer._raw.items():  # noqa: SLF001
            reader._raw[key] = blob  # noqa: SLF001
        reader._raw_index[writer_cache.keys["index"]] = set(  # noqa: SLF001
            writer._raw_index[writer_cache.keys["index"]]  # noqa: SLF001
        )
    else:
        reader = RedisStore(name="reader-pickle", client=writer._store, tracer=reader_tracer)  # noqa: SLF001

    rebuilt, _, rebuild_log = DiffTreeStore(reader, tree_id="cross2").load(protocol=PROTOCOL_PICKLE)
    assert [outcome for _, _, outcome in rebuild_log] == [RESULT_CROSS] * 3
    _assert_tree_shape(rebuilt)
    # Bytes remain JSON-owned after the pickle-first attempt falls back.
    for key in order:
        _, used = decode_blob(reader.get_raw(key))
        assert used == PROTOCOL_JSON


def test_decode_pickle_first_then_json_fallback():
    pickled = encode_blob({"type": "site", "children": []}, PROTOCOL_PICKLE)
    jsoned = encode_blob({"type": "site", "children": []}, PROTOCOL_JSON)
    assert decode_blob(pickled)[1] == PROTOCOL_PICKLE
    assert decode_blob(jsoned)[1] == PROTOCOL_JSON
    with pytest.raises(ValueError):
        decode_blob(b"not a pickle nor json \x80\x81")


def test_fake_redis_nx_first_writer_wins():
    client = FakeRedis()
    assert client.set("k", b"first", nx=True) is True
    assert client.set("k", b"second", nx=True) is False
    assert client.get("k") == b"first"
    client.sadd("s", "a", "a", "b")
    assert client.smembers("s") == {b"a", b"b"}
