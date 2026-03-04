import hypothesis.strategies as st
from conftest import cylinders, planes, spheres
from hypothesis import given, settings

from coremaker.surfaces.surfacecache import SurfaceCache


@settings(deadline=None)
@given(st.one_of(cylinders, spheres, planes),
       st.one_of(cylinders, spheres, planes))
def test_surface_cache_detect_isclose(s1, s2):
    cache = SurfaceCache(lambda x: 0)
    ind, s = cache.find_surface(s1, 0)
    assert ind in (1, -1)
    if s1.isclose(s2):
        assert cache.find_surface(s2, 0)[0] == ind
    elif s1.isclose(-s2):
        assert cache.find_surface(s2, 0)[0] == -ind
    else:
        assert cache.find_surface(s2, 0)[0] in {-2, 2}
