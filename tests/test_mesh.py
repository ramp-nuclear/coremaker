"""Tests for Mesh objects"""

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from coremaker.mesh import CartesianMesh

points = arrays(dtype=float, shape=3, elements=st.floats(min_value=-10, max_value=10, allow_subnormal=False))
positives = arrays(dtype=float, shape=3, elements=st.floats(min_value=1e-5, max_value=10, allow_subnormal=False))
pairs = st.tuples(points, positives).map(lambda x: (x[0], x[0] + x[1]))


@settings(deadline=None)
@given(pairs, positives.map(tuple))
def test_cartesian_mesh_from_vertices_minimal_maximal_boundaries_as_expected(vertices, resolution):
    lower, upper = vertices
    mesh = CartesianMesh.from_vertices(lower, upper, resolution=resolution)
    for low, high, attr in zip(lower, upper, "xyz"):
        assert getattr(mesh, attr)[0] == low
        assert getattr(mesh, attr)[-1] == high

