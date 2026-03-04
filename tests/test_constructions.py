"""Tests for Tree object elements.

"""
from functools import partial
from pathlib import PurePath

import pytest

from coremaker.elements.assembly import singular_root_construction
from coremaker.elements.box import BoxTree, ExcludeFrame, FrameBox, SplitBox, excludeframe_to_framebox
from coremaker.geometries.box import Box
from coremaker.materials.aluminium import al6061
from coremaker.materials.steel import steel_304L
from coremaker.transform import Transform
from coremaker.tree import ChildType

example_splitbox_dim = (1., 2., 3.)
example_inner_dim = (0.5, 0.5, 0.5)


def test_boxtree_has_one_known_component_by_example():
    dim = (1., 2., 3.)
    element = BoxTree(dim, al6061, PurePath('moo'))
    paths, components = zip(*element.named_components())
    assert paths == (PurePath('moo'),)
    assert len(components) == 1
    comp = components[0]
    assert comp.mixture == al6061
    assert comp.geometry == Box((0., 0., 0.), dim)


@pytest.fixture(scope='module')
def _example_splitbox() -> SplitBox:
    resolution = (0.5, 0.5, 0.5)
    return SplitBox(example_splitbox_dim, al6061, PurePath('moo'), resolution)


def test_splitbox_components_have_same_material_by_example(_example_splitbox):
    assert all(c.mixture == al6061 for c in _example_splitbox.components())


def test_splitbox_top_has_known_box_geometry_by_example(_example_splitbox):
    roots = tuple(_example_splitbox.roots())
    assert len(roots) == 1
    assert roots[0][0] == PurePath('moo')
    assert roots[0][1].geometry == Box((0., 0., 0.), example_splitbox_dim)


def test_splitbox_pieces_do_not_stray_from_outer_geo_by_example(_example_splitbox):
    geos = [c.geometry for c in _example_splitbox.components()]
    for geo in geos:
        surfaces = geo.surfaces
        for surf in surfaces:  # type: Plane
            a1, a2, a3 = surf.a
            match (abs(a1), abs(a2), abs(a3), surf.b):
                case (1., 0., 0., _):
                    val = example_splitbox_dim[0]/2
                case (0., 1., 0., _):
                    val = example_splitbox_dim[1]/2
                case (0., 0., 1., _):
                    val = example_splitbox_dim[2]/2
                case _:
                    raise ValueError("A plane was off and bad for use.")
            assert -val - 1e-4 <= surf.b <= val + 1e-4


@pytest.fixture(scope='module')
def _example_framebox() -> FrameBox:
    inner_dim = example_inner_dim
    outer_resolution = (0.5, 0.5, 0.5)
    inner_resolution = (0.1, 0.1, 0.1)
    return FrameBox(frame_dimensions=example_splitbox_dim,
                    picture_dimensions=inner_dim,
                    frame_name=PurePath('MooFrame'),
                    picture_name=PurePath('MooPicture'),
                    frame_resolution=outer_resolution,
                    picture_resolution=inner_resolution,
                    frame_mixture=steel_304L,
                    picture_mixture=al6061,
                    )


def test_framebox_top_has_known_box_geometry_by_example(_example_framebox):
    roots = tuple(_example_framebox.roots())
    assert len(roots) == 1
    assert roots[0][0] == PurePath('MooFrame')
    assert roots[0][1].geometry == Box((0., 0., 0.), example_splitbox_dim)


def test_framebox_pieces_have_correct_mixtures_by_example(_example_framebox):
    frames = filter(lambda x: 'FramePiece' in str(x[0]),
                    _example_framebox.named_components())
    pictures = filter(lambda x: 'MooFrame/MooPicture/' in str(x[0]),
                      _example_framebox.named_components())
    assert all(comp.mixture == steel_304L for _, comp in frames)
    assert all(comp.mixture == al6061 for _, comp in pictures)


def test_framebox_subdicts_all_start_at_root(_example_framebox):
    root = PurePath('MooFrame')
    for d in (_example_framebox.inclusive, _example_framebox.exclusive,
              _example_framebox.external_exclusive):
        for p, lst in d.items():
            assert root in p.parents or p == root, (p, list(p.parents))
            for p2, _ in lst:
                assert root in p2.parents or p2 == root, (p2, list(p2.parents))


@pytest.fixture(scope='module')
def _example_excludeframe() -> FrameBox:
    inner_dim = example_inner_dim
    inner_resolution = (0.1, 0.1, 0.1)
    return ExcludeFrame(frame_dimensions=example_splitbox_dim,
                        picture_dimensions=inner_dim,
                        frame_name=PurePath('MooFrame'),
                        picture_name=PurePath('MooPicture'),
                        picture_resolution=inner_resolution,
                        frame_mixture=steel_304L,
                        picture_mixture=al6061,
                        )


def test_framebox_picture_has_known_box_geometry_by_example(_example_framebox,
                                                            _example_excludeframe):
    for tree in (_example_excludeframe, _example_framebox):
        picture = tree[PurePath('MooFrame') / 'MooPicture']
        assert picture.geometry == Box((0., 0., 0.), example_inner_dim)


def test_excludebox_pieces_have_correct_mixtures_by_example(_example_excludeframe):
    frames = filter(lambda x: 'MooFrame' == str(x[0]),
                    _example_excludeframe.named_components())
    pictures = filter(lambda x: 'MooFrame/MooPicture/' in str(x[0]),
                      _example_excludeframe.named_components())
    assert all(comp.mixture == steel_304L for _, comp in frames)
    assert all(comp.mixture == al6061 for _, comp in pictures)


def test_boxtree_after_rename_has_no_original_name():
    dim = (1., 2., 3.)
    element = BoxTree(dim, al6061, PurePath('moo'))
    element.rename(PurePath('moo'), PurePath('woof'))
    assert PurePath('moo') not in element
    assert PurePath('woof') in element


def test_singular_construction_has_correct_suffixes_by_example():
    dim = (1., 2., 3.)
    factory = partial(BoxTree, dim, al6061, PurePath('moo'))
    transforms = 3*[Transform()]
    suffixes = list(map(str, range(3)))
    tree = singular_root_construction(tuple(zip(3*[factory], transforms, suffixes)),
                                      root_path=PurePath('root'),
                                      relationship=ChildType.inclusive
                                      )
    for path in [PurePath('root') / PurePath(f'moo_{suffix}') for suffix in suffixes]:
        assert path in tree


def test_exclude_to_frame_makes_an_equivalent_frame_by_example():
    inner_dim = example_inner_dim
    inner_resolution = (0.1, 0.1, 0.1)
    outer_resolution = (0.5, 0.5, 0.5)
    exclude = ExcludeFrame(frame_dimensions=example_splitbox_dim,
                           picture_dimensions=inner_dim,
                           frame_name=PurePath('MooFrame'),
                           picture_name=PurePath('MooPicture'),
                           picture_resolution=inner_resolution,
                           frame_mixture=steel_304L,
                           picture_mixture=al6061,
                           )
    framebox = FrameBox(frame_dimensions=example_splitbox_dim,
                        picture_dimensions=inner_dim,
                        frame_name=PurePath('MooFrame'),
                        picture_name=PurePath('MooPicture'),
                        frame_resolution=outer_resolution,
                        picture_resolution=inner_resolution,
                        frame_mixture=steel_304L,
                        picture_mixture=al6061,
                        )
    trans_framebox = excludeframe_to_framebox(exclude,
                                              picture_name=PurePath('MooPicture'),
                                              frame_resolution=outer_resolution)
    assert trans_framebox == framebox
