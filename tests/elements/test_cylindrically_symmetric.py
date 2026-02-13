from itertools import pairwise
from pathlib import PurePath

import numpy as np

from coremaker.elements.cylindrically_symmetric import (
    radial_split,
    axial_split, _piece_centers, ChunkedCylinderTree,
    ChunkedAnnulusTree, FiniteCylinder, appropriate_radial_resolution,
    _splits_num, appropriate_axial_resolution
)
from coremaker.elements.util import split
from coremaker.geometries.annulus import Annulus
from coremaker.materials.aluminium import al1050
from coremaker.tree import Tree

LENGTH = 30.
MIXTURE = al1050
ROOT = PurePath('root')
AXIS = (0., 0., 1.)
LENGTHS = axial_split(LENGTH, LENGTH)
INNER_RADIUS = 5.
OUTER_RADIUS = 10.


def test_radial_split_for_example_case():
    radius = 20.
    radial_splitting = radial_split(radius, inner_radius=INNER_RADIUS)
    assert len(radial_splitting) == 2
    assert radial_splitting[0] == INNER_RADIUS
    assert radial_splitting[1] == radius
    resolution = radius - INNER_RADIUS
    radial_splitting = radial_split(radius, resolution=resolution,
                                    inner_radius=INNER_RADIUS)
    assert len(radial_splitting) == 3
    assert radial_splitting[0] == INNER_RADIUS
    assert np.isclose(radial_splitting[1] ** 2 - radial_splitting[0] ** 2,
                      radial_splitting[2] ** 2 - radial_splitting[1] ** 2)
    assert radial_splitting[-1] == radius


def test_axial_split_for_example_case():
    assert len(LENGTHS) == 2
    assert all(np.isclose(h, next_h) for h, next_h in pairwise(LENGTHS))


def test_piece_centers_for_example_case():
    length = sum(LENGTHS)
    piece_centers = _piece_centers(LENGTHS, (0., 0., 80.))
    assert np.isclose(piece_centers[0], [0., 0., -length / 4]).all()
    assert np.isclose(piece_centers[1], [0., 0., length / 4]).all()


def test_chunked_annulus_tree_for_example_case():
    resolution = (OUTER_RADIUS - INNER_RADIUS, LENGTH)
    chunked_annulus_tree = ChunkedAnnulusTree(inner_radius=INNER_RADIUS,
                                              outer_radius=OUTER_RADIUS,
                                              axis=AXIS,
                                              mixture=MIXTURE,
                                              root_path=ROOT,
                                              length=LENGTH,
                                              resolution=resolution)
    assert isinstance(chunked_annulus_tree.nodes[ROOT].geometry, Annulus)
    assert len(chunked_annulus_tree.nodes) == 5
    assert all(isinstance(node.geometry, Annulus)
               for node in chunked_annulus_tree.nodes.values())


def test_chunked_cylinder_for_example_case():
    resolution = (INNER_RADIUS, LENGTH)
    chunked_cylinder = ChunkedCylinderTree(INNER_RADIUS,
                                           LENGTH,
                                           MIXTURE,
                                           ROOT,
                                           AXIS,
                                           resolution)
    assert isinstance(chunked_cylinder, Tree)
    assert isinstance(chunked_cylinder.outer_geometry, FiniteCylinder)
    assert len(chunked_cylinder.nodes) == 5
    cylinder_nodes = {path: node
                      for (path, node) in chunked_cylinder.nodes.items()
                      if isinstance(node.geometry, FiniteCylinder)}
    assert len(cylinder_nodes) == 3


def test_appropriate_radial_resolution_against_splits_num():
    for n in range(2, 100):
        res: float = appropriate_radial_resolution(n, OUTER_RADIUS, INNER_RADIUS)
        i = _splits_num(OUTER_RADIUS, res, INNER_RADIUS)
        assert n == i


def test_appropriate_axial_resolution_against_split():
    for n in range(2, 100):
        res = appropriate_axial_resolution(n, LENGTH)
        i, = split([LENGTH], [res])
        assert i == n
