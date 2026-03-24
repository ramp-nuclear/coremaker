"""Testing tree tools"""

from copy import deepcopy
from pathlib import PurePath

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

from coremaker.elements.box import BoxTree, ExcludeFrame
from coremaker.geometries.box import Box
from coremaker.materials.aluminium import al1050
from coremaker.materials.steel import steel_304L
from coremaker.plane_intersection import intersect_tree
from coremaker.transform import Transform, identity, rotate180
from coremaker.tree import ChildType, Tree, _switch


def test_tree_default_name_is_none():
    assert Tree().name is None


def test_tree_name_attribute():
    t = Tree(name="Fuel-1")
    assert t.name == "Fuel-1"


def test_tree_serialize_includes_name():
    t = Tree(name="MyRod")
    _, d = t.serialize()
    assert d["name"] == "MyRod"


def test_tree_serialize_none_name():
    t = Tree()
    _, d = t.serialize()
    assert d["name"] is None


def test_tree_deserialize_without_name_key():
    """Old serialized data without a 'name' key should deserialize with name=None."""
    d = {"nodes": {}, "inclusive": {}, "exclusive": {}, "external_exclusive": {}}
    t = Tree.deserialize(d, supported={})
    assert t.name is None


def test_tree_equality_considers_name():
    t1 = Tree(name="A")
    t2 = Tree(name="B")
    t3 = Tree(name="A")
    assert t1 != t2
    assert t1 == t3


def test_tree_equality_named_vs_unnamed():
    t1 = Tree(name="A")
    t2 = Tree()
    assert t1 != t2


@pytest.mark.parametrize(
    ["p", "old", "new", "res"],
    [
        ("moo/foo", "moo", "woof", "woof/foo"),
        ("woof", "moo", "baaa", "woof"),
        ("moo", "moo", "woof", "woof"),
        ("baaa/meeee/moo/waaa", "baaa/meeee/moo", "gaga", "gaga/waaa"),
    ],
)
def test_switch_by_examples(p, old, new, res):
    p, old, new, res = map(PurePath, (p, old, new, res))
    assert _switch(p, old, new) == res


example_frame = ExcludeFrame(
    frame_dimensions=(3.0, 3.0, 3.0),
    picture_dimensions=(1.0, 1.0, 1.0),
    frame_name=PurePath("frame"),
    picture_name=PurePath("picture"),
    frame_mixture=steel_304L,
    picture_mixture=al1050,
)


def test_graft_computes_with_plane_intersection_by_example():
    t, tcopy = (deepcopy(example_frame) for _ in range(2))
    t2 = BoxTree((5.0, 5.0, 5.0), al1050, PurePath("super"))
    t3 = deepcopy(t)
    t4 = deepcopy(t2)
    z0 = 2
    t2.graft(t, PurePath("super"), ChildType.exclusive)
    intersect_of_graft = intersect_tree(t2, z0)
    graft_of_intersect = intersect_tree(t4, z0)
    graft_of_intersect.graft(intersect_tree(t3, z0), PurePath("super"), ChildType.exclusive)
    assert intersect_of_graft == graft_of_intersect


def test_subtree_of_graft_is_branch_by_example():
    t, tcopy = (deepcopy(example_frame) for _ in range(2))
    t2 = BoxTree((5.0, 5.0, 5.0), al1050, PurePath("super"))
    t2.graft(t, PurePath("super"), ChildType.exclusive)
    sub = t2.subtree(PurePath("super/frame"))
    assert sub == tcopy


def test_cut_of_graft_is_original_by_example():
    t = deepcopy(example_frame)
    t2 = BoxTree((5.0, 5.0, 5.0), al1050, PurePath("super"))
    t2.graft(t, PurePath("super"), ChildType.exclusive)
    t2.cut(PurePath("super/frame"))
    assert t2 == BoxTree((5.0, 5.0, 5.0), al1050, PurePath("super"))


def test_cut_empty():
    t = BoxTree((5.0, 5.0, 5.0), al1050, PurePath("super"))
    t.cut(PurePath("super"))
    assert t == Tree()


shiftframe = ExcludeFrame(
    frame_dimensions=(3.0, 3.0, 3.0),
    picture_dimensions=(1.0, 1.0, 1.0),
    frame_name=PurePath("frame"),
    picture_name=PurePath("picture"),
    frame_mixture=steel_304L,
    picture_mixture=al1050,
    picture_translation=(0.5, 0.0, 0.0),
)


shifts = arrays(float, 3, elements=st.floats(min_value=-1, max_value=1))
rotvecs = st.floats(min_value=0, max_value=2 * np.pi).map(lambda x: (0.0, 0.0, x))
transforms = st.builds(Transform.from_rotvec, shifts, rotvecs)


@given(shift=shifts, transform=transforms)
def test_relative_tree_transform_absolute_transforms_absolutely(shift, transform):
    tree = deepcopy(shiftframe)
    cur = tree.get_transform(PurePath("frame/picture"))
    tree.transform(PurePath("frame"), transform)
    tshift = Transform(translation=tuple(shift))
    tree.transform(PurePath("frame/picture"), tshift, relative_to=identity)
    finish = tree.get_transform(PurePath("frame/picture"))
    assert np.allclose((finish.translation - cur.translation).flatten(), shift, rtol=1e-6, atol=1e-6)


def test_shift_tree_multiple_roots_with_relative_shifts_all():
    shft = (1.0, 1.0, 0.0)
    tree = deepcopy(shiftframe)
    otree = deepcopy(shiftframe)
    tree.nodes[PurePath("foo")] = otree[PurePath("frame")]
    tree.nodes[PurePath("foo/bar")] = otree[PurePath("frame/picture")]
    tree.exclusive[PurePath("foo")] = [(PurePath("foo/bar"), otree[PurePath("frame/picture")])]
    tree.transform(None, Transform(translation=shft), relative_to=identity)
    assert tree.get_transform(PurePath("foo")) == tree.get_transform(PurePath("frame"))
    assert tree.get_transform(PurePath("foo/bar")) == tree.get_transform(PurePath("frame/picture"))
    shift = tree.get_transform(PurePath("frame")).translation - shiftframe.get_transform(PurePath("frame")).translation
    assert np.allclose(shift.flatten(), np.array(shft))
    shift = (
        tree.get_transform(PurePath("frame/picture")).translation
        - shiftframe.get_transform(PurePath("frame/picture")).translation
    )
    assert np.allclose(shift.flatten(), np.array(shft))


@pytest.mark.parametrize(
    ("tree", "path", "geo", "tree_transform"),
    [
        (example_frame, PurePath("frame"), Box((0, 0, 0), (3.0, 3.0, 3.0)), identity),
        (example_frame, PurePath("frame/picture"), Box((0, 0, 0), (1.0, 1.0, 1.0)), Transform((10.0, 0.0, 0.0))),
        (shiftframe, PurePath("frame"), Box((0, 0, 0), (3.0, 3.0, 3.0)), rotate180),
    ],
)
def test_get_geometry_by_example(tree, path, geo, tree_transform):
    tree.transform(None, tree_transform)
    assert tree.geometry_of(path) == geo.transform(tree_transform)
