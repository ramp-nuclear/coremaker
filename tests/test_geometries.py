"""Tests for different geometries

"""
from copy import copy
from math import pi

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from scipy.linalg import norm as norm2

from conftest import balls, boxes, hexprisms, rectangles, rings, circles
from conftest import posfloats, annuli, transforms, finitecylinders, medfloats
from coremaker.geometries.annulus import Annulus, Ring
from coremaker.geometries.ball import Ball, Circle
from coremaker.geometries.bare import BareGeometry
from coremaker.geometries.box import Box, Rectangle
from coremaker.geometries.cylinder import FiniteCylinder
from coremaker.geometries.hex import HexPrism
from coremaker.geometries.infinite import infiniteGeometry
from coremaker.plane_intersection import intersect_geometry
from coremaker.protocols.geometry import Geometry
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.sphere import Sphere
from coremaker.transform import Transform, counterclockwise_90deg


def test_infinite_geometry_has_no_surfaces():
    assert infiniteGeometry.surfaces == ()


ORIGIN = (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    'geo',
    [Annulus(ORIGIN, 1., 2., 1., (0.0, 0.0, 1.0)),
     Ball(ORIGIN, 1.),
     BareGeometry([]),
     Box(ORIGIN, np.ones(3, dtype=float)),
     FiniteCylinder(ORIGIN, 1., 1., (0.0, 0.0, 1.0)),
     HexPrism(ORIGIN, 1., 1.),
     infiniteGeometry,
     Rectangle(ORIGIN[:2], (1.,) * 2),
     Ring(ORIGIN[:2], 1, 2),
     Circle(ORIGIN[:2], 1)
     ])
def test_example_geometry_is_considered_a_geometry(geo):
    assert isinstance(geo, Geometry), type(geo)


@given(posfloats, posfloats)
def test_annulus_defends_against_illegal_radii(inner, dr):
    with pytest.raises(ValueError):
        Annulus((0.0, 0.0, 0.0), inner, inner - dr, 1., (0.0, 0.0, 1.0))


@given(st.one_of(finitecylinders, annuli), st.floats(0.0, 2 * pi))
def test_cylinder_annulus_rotation_around_axis_is_meaningless(s, angle):
    axis = np.array(s.axis)
    axis = axis / norm2(axis, ord=2)
    center = np.array(s.center)
    t = Transform.from_rotvec(s.center, axis * angle) @ Transform(-center)
    assert s == s.transform(t)


@given(annuli, transforms)
def test_transform_of_annulus_does_not_change_its_dimensions(annulus, trans):
    rotated = annulus.transform(trans)
    assert all(np.isclose(getattr(annulus, d),
                          getattr(rotated, d),
                          atol=1e-10, rtol=1e-10)
               for d in ['inner_radius', 'outer_radius', 'length'])


@given(st.one_of(finitecylinders, annuli), medfloats.filter(lambda x: abs(x) > 1e-3))
def test_axis_of_cylinder_annulus_is_same_upto_different_axis_length(s, factor):
    s2 = copy(s)
    s2.axis = tuple(factor * x for x in s2.axis)
    assert s == s2


@given(finitecylinders, transforms)
def test_transform_of_cylinder_does_not_change_its_dimensions(cylinder, trans):
    rotated = cylinder.transform(trans)
    assert all(np.isclose(getattr(cylinder, d),
                          getattr(rotated, d),
                          atol=1e-10, rtol=1e-10)
               for d in ['radius', 'length'])


# TODO: Add a test for HexPrism
@pytest.mark.parametrize(
    ('geo', 'surfaces'),
    [(Annulus((1, 0, 0), 1., 2., 4., (0.0, 0.0, 1.0)),
      (Plane(0, 0, 1, -2), -Plane(0, 0, 1, 2),
       Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=False),
       Cylinder((1., 0.0, 0.0), 2., (0.0, 0.0, 1.0), inside=True))),
     (Ball((1, 0, 0), 3.),
      (Sphere((1.0, 0.0, 0.0), 3., inside=True),)),
     (Box(np.array([1, 0, 0]), np.array([1, 2, 3])),
      (Plane(1, 0, 0, 0.5), -Plane(1, 0, 0, 1.5),
       Plane(0, 1, 0, -1), -Plane(0, 1, 0, 1),
       Plane(0, 0, 1, -1.5), -Plane(0, 0, 1, 1.5))),
     (Box(np.zeros(3), np.array([1, 2, 3]),
          Transform((1., 0., 0.)) @ counterclockwise_90deg),
      (Plane(1, 0, 0, 0), -Plane(1, 0, 0, 2),
       Plane(0, 1, 0, -0.5), -Plane(0, 1, 0, 0.5),
       Plane(0, 0, 1, -1.5), -Plane(0, 0, 1, 1.5))),
     (FiniteCylinder((1.0, 0.0, 0.0), 1., 1., (0.0, 0.0, 1.0)),
      (Plane(0, 0, 1, -0.5), -Plane(0, 0, 1, 0.5),
       Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=True))),
     (infiniteGeometry, tuple()),
     (Circle((1, 0), 3.), (Cylinder((1.0, 0, 0), 3., (0, 0, 1), inside=True),)),
     (Rectangle(np.array([1, 0]), np.array([1, 2])),
      (Plane(1, 0, 0, 0.5), -Plane(1, 0, 0, 1.5),
       Plane(0, 1, 0, -1), -Plane(0, 1, 0, 1))),
     (Ring((1, 0), 1., 2.),
      (Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=False),
       Cylinder((1., 0.0, 0.0), 2., (0.0, 0.0, 1.0), inside=True))),
     ]
)
def test_geometry_surfaces_by_example(geo, surfaces):
    assert sum(s1.isclose(s2)
               for s1 in geo.surfaces for s2 in surfaces) == len(surfaces)


@given(st.one_of(balls, boxes, finitecylinders, annuli, hexprisms, rectangles, circles, rings))
def test_geometry_surfaces_hashable(geom: Geometry):
    assert frozenset(geom.surfaces)


@pytest.mark.parametrize(
    ('geo'),
    [Annulus((1, 0, 0), 1., 2., 4., (0, 0, 1)),
     Ball(np.array([1, 0, 0]), 3.),
     Box(np.array([1, 0, 0]), np.array([1, 2, 3])),
     Box(np.zeros(3), np.array([1, 2, 3])),
     FiniteCylinder((1, 0, 0), 1., 1., (0, 0, 1))])
def test_intersection_of_geometry_and_plane_is_empty_when_it_should_be(geo):
    assert intersect_geometry(geo, 0.3)
    assert intersect_geometry(geo, 10) is None
