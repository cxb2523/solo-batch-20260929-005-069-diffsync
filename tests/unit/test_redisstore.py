"""Testing of RedisStore with the in-memory Redis double."""

import pytest

from diffsync.exceptions import ObjectStoreException
from diffsync.store.redis import RedisStore
from trace_store import FakeRedis


@pytest.fixture
def redisdb():
    """Provide a fresh in-memory Redis implementation."""
    return FakeRedis()


def _store(redisdb):
    return RedisStore(name="mystore", store_id="123", client=redisdb)


def test_redisstore_init(redisdb):
    store = _store(redisdb)
    assert str(store) == "mystore (123)"


def test_redisstore_init_wrong():
    with pytest.raises(ObjectStoreException):
        RedisStore(name="mystore", store_id="123", url="redis://wrong")


def test_redisstore_add_obj(redisdb, make_site):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    assert store.count() == 1


def test_redisstore_add_obj_twice(redisdb, make_site):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    store.add(obj=site)
    assert store.count() == 1


def test_redisstore_get_all_obj(redisdb, make_site):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    assert store.get_all(model=site.__class__)[0] == site


def test_redisstore_get_obj(redisdb, make_site):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    assert store.get(model=site.__class__, identifier=site.name) == site


def test_redisstore_remove_obj(redisdb, make_site):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    assert store.count(model=site.__class__) == store.count() == 1
    store.remove(obj=site)
    assert store.count(model=site.__class__) == store.count() == 0


def test_redisstore_get_all_model_names(redisdb, make_site, make_device):
    store = _store(redisdb)
    site = make_site()
    store.add(obj=site)
    device = make_device()
    store.add(obj=device)
    assert site.get_type() in store.get_all_model_names()
    assert device.get_type() in store.get_all_model_names()
