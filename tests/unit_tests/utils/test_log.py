from maqcache.utils.log import maqcache_log


def test_error_type():
    maqcache_log.setLevel("INFO")
    maqcache_log.error("Cache log error.")
    maqcache_log.warning("Cache log warning.")
    maqcache_log.info("Cache log info.")
    assert maqcache_log.level == 20
