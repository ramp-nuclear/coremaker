from itertools import pairwise
from pathlib import PurePath

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from coremaker.elements.cylindrically_symmetric import (
    ChunkedAnnulusTree,
    ChunkedCylinderTree,
    FiniteCylinder,
    UnequallyChunkedAnnulusTree,
    UnequallyChunkedCylinderTree,
    _piece_centers,
    _splits_num,
    appropriate_axial_resolution,
    appropriate_radial_resolution,
    axial_split,
    radial_split,
)
from coremaker.elements.util import split
from coremaker.geometries.annulus import Annulus
from coremaker.materials.aluminium import al1050
from coremaker.tree import Tree

pos = st.shared(st.floats(min_value=1e-5, max_value=5), key="inner")


@given(pos, pos.flatmap(lambda x: st.floats(min_value=x + 1e-5, max_value=x + 5)))
def test_radial_split_with_no_resolution_has_inner_and_outer_only(inner_radius, radius):
    radial_splitting = radial_split(radius, inner_radius=inner_radius)
    assert len(radial_splitting) == 2
    assert radial_splitting[0] == inner_radius
    assert radial_splitting[1] == radius


@given(pos, pos.flatmap(lambda x: st.floats(min_value=x + 1e-5, max_value=x + 5)))
def test_radial_split_with_resolution_of_full_annulus_is_split_to_3_in_known_way(inner_radius, radius):
    resolution = radius - inner_radius
    radial_splitting = radial_split(radius, resolution=resolution, inner_radius=inner_radius)
    assert len(radial_splitting) == 3
    assert radial_splitting[0] == inner_radius
    assert np.isclose(radial_splitting[1] ** 2 - radial_splitting[0] ** 2,
                      radial_splitting[2] ** 2 - radial_splitting[1] ** 2)
    assert radial_splitting[-1] == radius


def test_axial_split_gives_similar_chunks_for_equal_numbers_in_one_example():
    lengths = axial_split(30., 30.)
    assert len(lengths) == 2
    assert all(np.isclose(h, next_h) for h, next_h in pairwise(lengths))


def test_piece_centers_for_one_example_is_as_known():
    lengths = axial_split(30., 30.)
    length = sum(lengths)
    piece_centers = _piece_centers(lengths, (0., 0., 80.))
    assert np.isclose(piece_centers[0], [0., 0., -length / 4]).all()
    assert np.isclose(piece_centers[1], [0., 0., length / 4]).all()


ROOT_PATH = PurePath("root")
ZAXIS = (0., 0., 1.)


def test_chunked_annulus_tree_for_known_values_gives_known_tree_structure():
    inner_radius, outer_radius, length = 5., 10., 30.
    resolution = (outer_radius - inner_radius, length)
    chunked_annulus_tree = ChunkedAnnulusTree(inner_radius=inner_radius,
                                              outer_radius=outer_radius,
                                              axis=ZAXIS,
                                              mixture=al1050,
                                              root_path=ROOT_PATH,
                                              length=length,
                                              resolution=resolution)
    assert isinstance(chunked_annulus_tree.nodes[ROOT_PATH].geometry, Annulus)
    assert len(chunked_annulus_tree.nodes) == 5
    assert all(isinstance(node.geometry, Annulus)
               for node in chunked_annulus_tree.nodes.values())


def test_chunked_cylinder_tree_for_known_values_gives_known_tree_structure():
    radius, length = 5., 30.
    resolution = (radius, length)
    chunked_cylinder = ChunkedCylinderTree(radius,
                                           length,
                                           al1050,
                                           ROOT_PATH,
                                           ZAXIS,
                                           resolution)
    assert isinstance(chunked_cylinder, Tree)
    assert isinstance(chunked_cylinder.outer_geometry, FiniteCylinder)
    assert len(chunked_cylinder.nodes) == 5
    cylinder_nodes = {path: node
                      for (path, node) in chunked_cylinder.nodes.items()
                      if isinstance(node.geometry, FiniteCylinder)}
    assert len(cylinder_nodes) == 3


def test_appropriate_radial_resolution_splits_annular_ring_according_with_splits_num():
    inner_radius, outer_radius = 5., 10.
    for n in range(2, 100):
        res: float = appropriate_radial_resolution(n, outer_radius, inner_radius)
        i = _splits_num(outer_radius, res, inner_radius)
        assert n == i


def test_appropriate_axial_resolution_for_single_length():
    length = 30.
    for n in range(2, 100):
        res = appropriate_axial_resolution(n, length)
        i, = split([length], [res])
        assert i == n


def test_unequally_chunked_annulus_tree_for_known_values_gives_known_tree_structure():
    inner_radius, outer_radius, length = 5., 10., 30.
    lengths = [length / 3] * 3
    chunked_annulus_tree = UnequallyChunkedAnnulusTree(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            axis=ZAXIS,
            mixture=al1050,
            root_path=ROOT_PATH,
            length=length,
            lengths=lengths)
    assert isinstance(chunked_annulus_tree.nodes[ROOT_PATH].geometry, Annulus)
    assert len(chunked_annulus_tree.nodes) == 4
    assert all(isinstance(node.geometry, Annulus)
               for node in chunked_annulus_tree.nodes.values())


def test_unequally_chunked_cylinder_tree_for_known_values_gives_known_tree_structure():
    radius, length = 5., 30.
    lengths = [length / 3] * 3
    chunked_cylinder = UnequallyChunkedCylinderTree(
            radius=radius,
            axis=ZAXIS,
            mixture=al1050,
            root_path=ROOT_PATH,
            length=length,
            lengths=lengths)
    assert isinstance(chunked_cylinder, Tree)
    assert isinstance(chunked_cylinder.outer_geometry, FiniteCylinder)
    assert len(chunked_cylinder.nodes) == 4
    cylinder_nodes = {path: node for path, node in chunked_cylinder.nodes.items()
                      if isinstance(node.geometry, FiniteCylinder)}
    assert len(cylinder_nodes) == 4


lengths_list = st.shared(arrays(int, 3, elements=st.integers(min_value=1, max_value=5)), key="zlist")
cyls = lengths_list.map(lambda x: UnequallyChunkedCylinderTree(
    radius=5,
    length=np.sum(x).item(),
    mixture=al1050,
    root_path=PurePath("rod"),
    axis=ZAXIS,
    lengths=x,
    ))
annuli = lengths_list.map(lambda x: UnequallyChunkedAnnulusTree(
    outer_radius=5,
    inner_radius=2,
    length=np.sum(x).item(),
    mixture=al1050,
    root_path=PurePath("rod"),
    axis=ZAXIS,
    lengths=x
    ))


@given(lengths=lengths_list, tree=cyls | annuli)
def test_unequally_chunked_cylinder_tree_centers_are_where_they_are_expected(lengths: np.ndarray, tree: Tree):
    length = np.sum(lengths).item()
    z = sorted([node.transform.translation[2][0] for node in tree.nodes.values()][1:])
    cum_lengths = np.concatenate(([0], np.cumsum(lengths[:-1])))
    approximate_centers = (cum_lengths + lengths / 2 - length / 2)
    assert np.allclose(z, approximate_centers)

