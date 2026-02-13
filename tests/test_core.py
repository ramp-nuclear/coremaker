"""Tests on the core object

"""
from pathlib import PurePath

import pytest

from coremaker.core import Core
from coremaker.elements.box import BoxTree
from coremaker.grid import CartesianGrid
from coremaker.materials.aluminium import al1050
from coremaker.plane_intersection import intersect_core, intersect_tree
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
