from pathlib import PurePath

from coremaker.elements.box import BoxTree, split_box_inside_tree, SplitBox, ExcludeFrame, \
    excludeframe_to_framebox
from coremaker.materials.aluminium import al6061


def test_split_box_same_as_splitbox():
    box = BoxTree((1, 1, 1), al6061, PurePath("Box"))
    split_box_inside_tree(box, box_path=PurePath("Box"), resolution=(0.1, 0.1, 0.1))
    split_box = SplitBox((1, 1, 1), al6061, PurePath("Box"), (0.1, 0.1, 0.1))
    assert box.nodes.keys() == split_box.nodes.keys()
    assert box.exclusive.keys() == split_box.exclusive.keys()
    assert box.inclusive.keys() == split_box.inclusive.keys()
    assert box.external_exclusive.keys() == split_box.external_exclusive.keys()


def test_split_excludedframe_same_as_framebox():
    box1 = ExcludeFrame(frame_name=PurePath("Frame"), picture_name=PurePath("Picture"),
                        frame_mixture=al6061, frame_dimensions=(1, 1, 1), picture_mixture=al6061,
                        picture_dimensions=(0.5, 0.5, 0.5), picture_translation=(0.1, 0.1, 0.1))
    box2 = ExcludeFrame(frame_name=PurePath("Frame"), picture_name=PurePath("Picture"),
                        frame_mixture=al6061, frame_dimensions=(1, 1, 1), picture_mixture=al6061,
                        picture_dimensions=(0.5, 0.5, 0.5), picture_translation=(0.1, 0.1, 0.1))
    split_box_inside_tree(box1, box_path=PurePath("Frame"), resolution=(0.1, 0.1, 0.1))
    split_box = excludeframe_to_framebox(box2, PurePath("Picture"), (0.1, 0.1, 0.1))
    assert len(set(split_box.nodes.keys()).symmetric_difference(box1.nodes.keys())) == 1
