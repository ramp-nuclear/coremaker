"""Tests for different geometries

"""
from copy import copy
from math import pi

import hypothesis.strategies as st
import numpy as np
import pytest
from conftest import (
    annuli,
    balls,
    boxes,
    circles,
    finitecylinders,
    hexprisms,
    medfloats,
    posfloats,
    rectangles,
    rings,
    transforms,
    translations,
)
from hypothesis import given
from scipy.linalg import norm as norm2

from coremaker.geometries import *
from coremaker.plane_intersection import intersect_geometry
from coremaker.protocols.geometry import Geometry
from coremaker.surfaces import Cylinder, Plane, Sphere
from coremaker.transform import Transform, counterclockwise_90deg


def test_infinite_geometry_has_no_surfaces():
    assert infiniteGeometry.surfaces == ()


ORIGIN = (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    'geo',
    [Annulus(ORIGIN, 1., 2., 1., (0.0, 0.0, 1.0)),
     Ball(ORIGIN, 1.),
     BareGeometry([]),
     Box(ORIGIN, (1., 1., 1.)),
     FiniteCylinder(ORIGIN, 1., 1., (0.0, 0.0, 1.0)),
     HexPrism(ORIGIN, 1., 1.),
     infiniteGeometry,
     Rectangle(ORIGIN[:2], (1.,) * 2),
     Ring(ORIGIN[:2], 1, 2),
     Circle(ORIGIN[:2], 1)
     ])
def test_example_geometry_is_considered_a_geometry(geo):
    assert isinstance(geo, Geometry), type(geo)


points = st.tuples(*(3 * [st.floats(min_value=-10, max_value=10, allow_subnormal=False)]))
positive_3d = st.tuples(*(3 * [st.floats(min_value=1e-6, max_value=10)]))
pairs = st.tuples(points, positive_3d).map(lambda x: (x[0], tuple(y1 + y2 for y1, y2 in zip(*x))))


@given(pairs)
def test_box_creation_from_vertices_same_as_dimensions_and_center(vertices):
    v1, v2 = vertices
    x1, x2 = map(np.asarray, vertices)
    dimensions = tuple(x2 - x1)
    center = tuple((x2 + x1) / 2)
    b1 = Box.from_vertices(v1, v2)
    b2 = Box(center, dimensions)
    assert b1 == b2


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


@given(hexprisms, translations)
def test_transform_of_hexprisms_does_not_change_their_dimensions(hexa, trans):
    translated = hexa.transform(trans)
    assert all(np.isclose(getattr(hexa, d), getattr(translated, d), atol=1e-10, rtol=1e-10)
               for d in ["pitch", "height"])


example_prism = HexPrism((0, 0, 0), 5, 2)
hexy = np.sqrt(3.) / 2


@given(points)
def test_hexprism_translation_changes_center_as_expected_for_origin_hex(translation):
    new_prism = example_prism.transform(Transform(translation=translation))
    assert new_prism.center == translation


@given(points)
def test_hexprism_raises_for_rotation(rotation):
    rot = Transform.from_rotvec(rotation)
    if np.all(rot.rotation.as_rotvec() == (0, 0, 0)):
        example_prism.transform(rot)
        return
    with pytest.raises(NotImplementedError):
        example_prism.transform(rot)


@pytest.mark.parametrize(
    ('geo', 'surfaces'),
    [(Annulus((1, 0, 0), 1., 2., 4., (0.0, 0.0, 1.0)),
      (Plane(0, 0, 1, -2), -Plane(0, 0, 1, 2),
       Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=False),
       Cylinder((1., 0.0, 0.0), 2., (0.0, 0.0, 1.0), inside=True))),
     (Ball((1, 0, 0), 3.),
      (Sphere((1.0, 0.0, 0.0), 3., inside=True),)),
     (Box((1, 0, 0), (1, 2, 3)),
      (Plane(1, 0, 0, 0.5), -Plane(1, 0, 0, 1.5),
       Plane(0, 1, 0, -1), -Plane(0, 1, 0, 1),
       Plane(0, 0, 1, -1.5), -Plane(0, 0, 1, 1.5))),
     (Box((0, 0, 0), (1, 2, 3),
          Transform((1., 0., 0.)) @ counterclockwise_90deg),
      (Plane(1, 0, 0, 0), -Plane(1, 0, 0, 2),
       Plane(0, 1, 0, -0.5), -Plane(0, 1, 0, 0.5),
       Plane(0, 0, 1, -1.5), -Plane(0, 0, 1, 1.5))),
     (FiniteCylinder((1.0, 0.0, 0.0), 1., 1., (0.0, 0.0, 1.0)),
      (Plane(0, 0, 1, -0.5), -Plane(0, 0, 1, 0.5),
       Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=True))),
     (infiniteGeometry, tuple()),
     (Circle((1, 0), 3.), (Cylinder((1.0, 0, 0), 3., (0, 0, 1), inside=True),)),
     (Rectangle((1, 0), (1, 2)),
      (Plane(1, 0, 0, 0.5), -Plane(1, 0, 0, 1.5),
       Plane(0, 1, 0, -1), -Plane(0, 1, 0, 1))),
     (Ring((1, 0), 1., 2.),
      (Cylinder((1., 0.0, 0.0), 1., (0.0, 0.0, 1.0), inside=False),
       Cylinder((1., 0.0, 0.0), 2., (0.0, 0.0, 1.0), inside=True))),
     (HexPrism((1., 0., 0.), 6., 2.),
      (Plane(1, 0, 0, -2), -Plane(1, 0, 0, 4),
       Plane(0.5, hexy, 0, -2.5), Plane(0.5, -hexy, 0, -2.5),
       -Plane(0.5, hexy, 0, 3.5), -Plane(0.5, -hexy, 0, 3.5),
       Plane(0, 0, 1, -1), -Plane(0, 0, 1, 1)
       )),
     ]
)
def test_geometry_surfaces_by_example(geo, surfaces):
    assert sum(s1.isclose(s2) for s1 in geo.surfaces for s2 in surfaces) == len(surfaces)


@given(st.one_of(balls, boxes, finitecylinders, annuli, hexprisms, rectangles, circles, rings))
def test_geometry_surfaces_hashable(geom: Geometry):
    assert frozenset(geom.surfaces)


@pytest.mark.parametrize(
    'geo',
    [Annulus((1, 0, 0), 1., 2., 4., (0, 0, 1)),
     Ball((1, 0, 0), 3.),
     Box((1, 0, 0), (1, 2, 3)),
     Box((0, 0, 0), (1, 2, 3)),
     FiniteCylinder((1, 0, 0), 1., 1., (0, 0, 1))])
def test_intersection_of_geometry_and_plane_is_empty_when_it_should_be(geo):
    assert intersect_geometry(geo, 0.3)
    assert intersect_geometry(geo, 10) is None
