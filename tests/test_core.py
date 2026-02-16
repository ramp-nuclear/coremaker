"""Tests on the core object

"""
from pathlib import PurePath

import numpy as np
import pytest

from coremaker.core import Core
from coremaker.elements.box import BoxTree
from coremaker.geometries import Box
from coremaker.grids import CartesianGrid
from coremaker.materials.aluminium import al1050
from coremaker.plane_intersection import intersect_core, intersect_tree
from coremaker.transform import Transform, rotate90
from coremaker.tree import Tree
from cytoolz import first


def test_fake_core_named_paths_start_with_site():
    simple = BoxTree((5., 5., 5.), al1050, PurePath('Box'))
    grid = CartesianGrid((0., 0., 0.), (2, 1), (1., 1.), 1., al1050,
                         rod_contents={'A1': simple})
    core_tree = Tree()
    core_tree.nodes[PurePath('lattice')] = first(grid.lattices)
    c = Core(grid, {}, core_tree)
    for name, comp in c.named_components:
        assert name.parents[-2] == PurePath('A1')


def test_fake_core_can_get_with_paths_from_named_component():
    simple = BoxTree((5., 5., 5.), al1050, PurePath('Box'))
    grid = CartesianGrid((0., 0., 0.), (2, 1), (1., 1.), 1., al1050,
                         rod_contents={'A1': simple})
    core_tree = Tree()
    core_tree.nodes[PurePath('lattice')] = first(grid.lattices)
    c = Core(grid, {}, core_tree)
    for name, _ in c.named_components:
        assert c[name]


def test_fake_core_raises_keyerror_if_key_is_unfound():
    simple = BoxTree((5., 5., 5.), al1050, PurePath('Box'))
    grid = CartesianGrid((0., 0., 0.), (2, 1), (1., 1.), 1., al1050,
                         rod_contents={'A1': simple})
    c = Core(grid, {}, simple)
    with pytest.raises(KeyError):
        assert c[PurePath('kakaka/rara')]


def test_intersect_core_is_like_intersecting_rods_outside_the_core():
    simple = BoxTree((5., 5., 5.), al1050, PurePath('Box'))
    grid = CartesianGrid((0., 0., 0.), (2, 1), (1., 1.), 1., al1050,
                         rod_contents={'A1': simple})
    c = intersect_core(Core(grid, {}, simple), 2)
    assert c.grid['A1'] == intersect_tree(simple, 2)


def test_geometry_is_given_in_reference_system_of_core_in_one_example():
    box_dimensions = (1., 4., 7.)
    box_translation = (2., 3., 8.)
    box_transform = Transform(box_translation)
    simple = BoxTree(box_dimensions, al1050, PurePath("Box"), transform=box_transform)
    center = (100., 50., 20.)
    grid = CartesianGrid(center, (1, 1), (10., 8.), 1., al1050, rod_contents={"A1": simple})
    core_tree = Tree()
    core_tree.nodes[PurePath("lattice")] = first(grid.lattices)
    core_tree.transform(None, rotate90)

    c = Core(grid, {}, core_tree)
    simple_path = PurePath("A1/Box")
    A1_transform = c.site_transform("A1")
    final_box_center = tuple(np.array(box_translation) + np.array(center))
    assert c.geometry_of(simple_path) == Box(final_box_center, box_dimensions, transform=rotate90)

