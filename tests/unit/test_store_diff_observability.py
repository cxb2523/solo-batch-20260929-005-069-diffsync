"""Tests for observable Diff storage in LocalStore and RedisStore."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from diffsync.diff import Diff, DiffElement
from diffsync.exceptions import ObjectNotFound
from diffsync.store import encode_diff_node_id
from diffsync.store.local import LocalStore
from diffsync.store.redis import RedisStore
from trace_store import FakeRedis


def make_three_level_diff() -> Diff:
    """Create a deterministic root-parent-child diff tree."""
    diff = Diff()
    root = DiffElement("region", "apac", {"name": "apac"})
    root.add_attrs(source={"owner": "net"}, dest={"owner": "sec"})
    parent = DiffElement("site", "sg", {"name": "sg"})
    parent.add_attrs(source={"status": "active"}, dest={"status": "maintenance"})
    child = DiffElement("device", "leaf-01", {"name": "leaf-01"})
    child.add_attrs(source={"role": "leaf"}, dest={"role": "spine"})
    parent.add_child(child)
    root.add_child(parent)
    diff.add(root)
    return diff


def make_element(name: str) -> DiffElement:
    """Create a simple source-only element."""
    element = DiffElement(name, name, {"name": name})
    element.add_attrs(source={})
    return element


@pytest.fixture
def local_store():
    """Provide a local store."""
    return LocalStore(name="local")


@pytest.fixture
def redis_store():
    """Provide a RedisStore backed by the in-memory Redis double."""
    return RedisStore(name="redis", store_id="unit-test", client=FakeRedis())


@pytest.mark.parametrize("store_fixture", ["local_store", "redis_store"])
def test_empty_diff_root_is_distinct_from_missing_root(request, store_fixture):
    """An empty root serializes as a present empty tree; a missing root remains a miss."""
    store = request.getfixturevalue(store_fixture)
    store.dump_diff(Diff(), tree="empty")
    store.trace_clear()

    rebuilt = store.load_diff(tree="empty")

    assert rebuilt.groups() == []
    assert store.last_diff_rebuild_order == []
    root_key = store._diff_root_key("empty")
    root_events = store.trace_key_events(root_key)
    assert [event.result for event in root_events if event.operation == "get"] == ["hit"]

    with pytest.raises(ObjectNotFound):
        store.load_diff(tree="missing")
    missing_events = store.trace_key_events(store._diff_root_key("missing"))
    assert [event.result for event in missing_events if event.operation == "get"] == ["miss"]


@pytest.mark.parametrize("store_fixture", ["local_store", "redis_store"])
def test_parent_and_child_concurrent_miss_deduplicates_same_key(request, store_fixture):
    """Parent and child both backfill once, while duplicate requests for the same key collapse."""
    if store_fixture == "redis_store":
        fake_redis = FakeRedis()
        store = RedisStore(name="redis-a", store_id="concurrency", client=fake_redis)
        peer = RedisStore(name="redis-b", store_id="concurrency", client=fake_redis)
    else:
        store = request.getfixturevalue(store_fixture)
        peer = store

    parent_id = encode_diff_node_id("parent", "p1")
    child_id = f"{parent_id}/{encode_diff_node_id('child', 'c1')}"
    calls = {"parent": 0, "child": 0}
    barrier = threading.Barrier(4)

    def load(active_store, level: str, node_id: str):
        barrier.wait()

        def factory():
            calls[level] += 1
            return make_element(level)

        element, _, result, backfilled = active_store.get_or_set_diff_node(
            tree="miss",
            node_id=node_id,
            factory=factory,
        )
        assert element.name == level
        return result, backfilled

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(
            executor.map(
                lambda args: load(*args),
                [
                    (store, "parent", parent_id),
                    (peer, "parent", parent_id),
                    (store, "child", child_id),
                    (peer, "child", child_id),
                ],
            )
        )

    assert calls == {"parent": 1, "child": 1}
    assert sorted(result for result, _ in outcomes) == ["backfill", "backfill", "hit", "hit"]

    parent_events = [
        event
        for active_store in (store, peer)
        for event in active_store.trace_key_events(active_store._diff_node_key("miss", parent_id))
    ]
    assert [event.result for event in parent_events if event.operation == "get"]
    assert any(event.result == "backfill" for event in parent_events)
    assert any(event.result == "hit" for event in parent_events)


@pytest.mark.parametrize("store_fixture", ["local_store", "redis_store"])
@pytest.mark.parametrize("written", ["json", "pickle"])
def test_cross_protocol_diff_node_read(request, store_fixture, written):
    """A node written by either protocol is read successfully by the JSON-first decoder."""
    store = request.getfixturevalue(store_fixture)
    node_id = encode_diff_node_id("cross", written)
    store.set_diff_node(tree="cross", node_id=node_id, element=make_element(written), protocol=written)

    read, observed, result, backfilled = store.get_or_set_diff_node(
        tree="cross",
        node_id=node_id,
        factory=lambda: pytest.fail("factory must not be called on a cross-protocol hit"),
        protocol="pickle" if written == "json" else "json",
    )

    assert read.name == written
    assert observed == written
    assert result == "hit"
    assert backfilled is False


@pytest.mark.parametrize("protocol", ["json", "pickle"])
def test_three_level_tree_round_trip_and_hit_rate(local_store, protocol):
    """Dump then rebuild follows preorder and all rebuild node reads hit."""
    source = make_three_level_diff()
    local_store.dump_diff(source, tree="tree", protocol=protocol)
    local_store.trace_clear()

    rebuilt = local_store.load_diff(tree="tree")

    assert local_store.last_diff_rebuild_order == [
        "region/apac",
        "region/apac/site/sg",
        "region/apac/site/sg/device/leaf-01",
    ]
    root = next(rebuilt.get_children())
    site = next(root.get_children())
    device = next(site.get_children())
    assert root.type == "region"
    assert site.type == "site"
    assert device.type == "device"
    summary = local_store.trace_summary()
    assert summary["gets"] == 5
    assert summary["hits"] == 5
    assert summary["hit_rate"] == 1.0
