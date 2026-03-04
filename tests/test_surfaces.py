"""Tests for the surface objects."""

import pickle
from math import pi, sqrt

import hypothesis.strategies as st
import numpy as np
import pytest
from conftest import cylinders, planes, spheres, transforms, translations
from hypothesis import given
from scipy.linalg import norm as norm2

from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.sphere import Sphere
from coremaker.surfaces.util import allclose
from coremaker.transform import Transform, counterclockwise_90deg


@given(translations, planes)
def test_translation_of_plane_changes_b_only(trans, p: Plane):
    newp = p.transform(trans)
    assert allclose(p.a, newp.a)


@given(planes)
def test_pickle_plane(p: Plane):
    assert pickle.loads(pickle.dumps(p)) == p


@pytest.mark.parametrize(
    ["p", "t", "res"],
    [
        (Plane(1, 0, 0, 0), counterclockwise_90deg, Plane(0, 1, 0, 0)),
        (Plane(1, 0, 0, 1), counterclockwise_90deg, Plane(0, 1, 0, 1)),
        (Plane(0, 1, 0, 0), counterclockwise_90deg, Plane(-1, 0, 0, 0)),
        (Plane(1, 0, 0, 0), counterclockwise_90deg @ counterclockwise_90deg, Plane(-1, 0, 0, 0)),
        (Plane(0, 1, 0, 0), counterclockwise_90deg @ counterclockwise_90deg, Plane(0, -1, 0, 0)),
        (
            Plane(0, 1, 0, 1),
            Transform.from_matrix(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
            Plane(0, -1, 0, 1),
        ),
        (Plane(1, 0, 0, -1), Transform((1.0, 0, 0)) @ counterclockwise_90deg, Plane(0, 1, 0, -1)),
    ],
)
def test_transform_of_plane_by_example(p, t, res):
    assert p.transform(t).isclose(res)


def test_zero_coefficients_are_illegal_for_plane():
    with pytest.raises(ValueError):
        Plane(0, 0, 0, 10, check=True)


def test_zero_coefficients_are_illegal_for_plane_normal():
    with pytest.raises(ValueError):
        Plane(0, 0, 0, 10).normal((0, 0, 0))


@given(cylinders, st.floats(0, 2 * pi))
def test_rotation_along_axis_does_nothing_for_cylinder(cyl: Cylinder, angle):
    axis = np.array(cyl.axis)
    axis = axis / norm2(axis, ord=2)
    center = np.array(cyl.center)
    t = Transform.from_rotvec(center, angle * axis) @ Transform(-center)
    assert cyl.transform(t).isclose(cyl)


@given(cylinders, translations)
def test_translation_of_cylinder_changes_center_only(c: Cylinder, t: Transform):
    nc = c.transform(t)
    assert (c.inside, c.radius, c.axis) == (nc.inside, nc.radius, nc.axis)


@pytest.mark.parametrize(
    ["cyl", "t", "res"],
    [
        (
            Cylinder((0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 1.0), inside=True),
            Transform(translation=np.ones(3)),
            Cylinder((1.0, 1.0, 0.0), 1.0, (0.0, 0.0, 1.0), inside=True),
        ),
        (
            Cylinder((0.0, 0.0, 0.0), 1.0, (0.0, 1.0, 0.0), inside=True),
            Transform.from_rotvec(rotation=(0, 0, np.pi / 2)),
            Cylinder((0.0, 0.0, 0.0), 1.0, (1.0, 0.0, 0.0), inside=True),
        ),
        (
            Cylinder((0.0, 0.0, 0.0), 1.0, (0.0, 1.0, 0.0), inside=True),
            Transform.from_rotvec(translation=np.ones(3), rotation=(0, 0, np.pi / 2)),
            Cylinder((0.0, 1.0, 1.0), 1.0, (1.0, 0.0, 0.0), inside=True),
        ),
    ],
)
def test_cylinder_transformation_by_example(cyl: Cylinder, t: Transform, res: Cylinder):
    assert cyl.transform(t).isclose(res)


@given(spheres, transforms)
def test_transform_on_sphere_only_changes_center(s: Sphere, t: Transform):
    ns = s.transform(t)
    assert s.radius == ns.radius
    assert s.inside == ns.inside


@given(st.one_of(planes, cylinders, spheres))
def test_double_neg_surface_is_surface(s):
    assert -(-s) == s


@pytest.mark.parametrize(
    ["surface", "point", "normal"],
    [
        (
            Plane(1, 0, 0, 0),
            (0, 5, 5),
            (-1, 0, 0),
        ),
        (
            Plane(1, 1, 0, 1),
            (1, 0, 0),
            (-1 / sqrt(2), -1 / sqrt(2), 0),
        ),
        (
            Plane(0, 0, -1, 0),
            (0, 0, 0),
            (0, 0, 1),
        ),
        (Sphere((0.0, 0.0, 0.0), 10.0, inside=False), (0, 0, 10), (0, 0, -1)),
        (Sphere((5.0, 0.0, 0.0), 10.0, inside=True), (5, 0, 10), (0, 0, 1)),
        (Cylinder((0.0, 0.0, 0.0), 5, axis=(0.0, 0.0, 0.1), inside=False), (5, 0, 0), (-1, 0, 0)),
        (Cylinder((9.0, 0.0, 0.0), 50, axis=(0.0, 1.0, 0.0), inside=True), (59.0, 0, 0), (1, 0, 0)),
    ],
)
def test_normal_of_surface_by_example(surface, point, normal):
    assert surface.normal(point) == normal
