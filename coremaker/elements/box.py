"""A tree based box element."""

import math
from itertools import pairwise, product
from pathlib import PurePath

import numpy as np

from coremaker.elements.util import split
from coremaker.geometries.box import Box
from coremaker.geometries.union import union_bounding_box
from coremaker.materials.mixture import Mixture
from coremaker.transform import Transform, identity
from coremaker.tree import ChildType, Node, Tree
from coremaker.units import cm

ORIGIN = (0.0, 0.0, 0.0)


def _framebox(
    *,
    picture: Tree,
    frame_dimensions: np.ndarray,
    picture_dimensions: np.ndarray,
    frame_name: PurePath,
    frame_resolution: tuple[cm, cm, cm] = (math.inf, math.inf, math.inf),
    frame_mixture: Mixture,
    picture_translation: np.ndarray = np.array((0.0, 0.0, 0.0)),
    transform: Transform = identity,
) -> Tree:
    element = Tree()
    element.nodes[frame_name] = Node(Box(ORIGIN, tuple(frame_dimensions)), transform=transform)
    element.graft(picture, frame_name, ChildType.inclusive)
    for sign_vec in map(lambda x: np.array(x, ndmin=1), filter(any, product([-1, 0, 1], repeat=3))):
        dimensions = (frame_dimensions - picture_dimensions) / 2 - sign_vec * picture_translation
        unmoved = np.abs(sign_vec) == 0
        dimensions[unmoved] = picture_dimensions[unmoved]
        if np.any(dimensions <= 0.0):
            continue
        center = picture_translation / 2 + sign_vec * (picture_dimensions + frame_dimensions) / 4
        center[unmoved] = picture_translation[unmoved]
        name = PurePath(f"FramePiece: {tuple(sign_vec)}")
        shift = Transform(translation=center)
        frame_piece = (
            SplitBox(
                dimensions=dimensions, mixture=frame_mixture, name=name, resolution=frame_resolution, transform=shift
            )
            if any(v > 1 for v in split(dimensions, frame_resolution))
            else BoxTree(dimensions=dimensions, mixture=frame_mixture, name=name, transform=shift)
        )
        element.graft(frame_piece, frame_name, ChildType.inclusive)
    return element


def FrameBox(
    *,
    frame_dimensions: tuple[cm, cm, cm],
    picture_dimensions: tuple[cm, cm, cm],
    frame_name: PurePath,
    picture_name: PurePath,
    frame_resolution: tuple[cm, cm, cm] = (math.inf, math.inf, math.inf),
    picture_resolution: tuple[cm, cm, cm] = (math.inf, math.inf, math.inf),
    frame_mixture: Mixture,
    picture_mixture: Mixture,
    picture_translation: tuple[cm, cm, cm] = (0.0, 0.0, 0.0),
    transform: Transform = identity,
) -> Tree:
    """Create a frame from one material containing an internal box (a picture,
    in the analogy this is framed as) made from another material. The picture
    can be set anywhere inside the frame and both of them can be split with
    a given resolution.
    The splitting on the internal box and the external frame may not align.
    The external frame is split automatically into 26 pieces, if the box is strictly
    contained within the frame. These 26 can be easily explained. In 2D, a FrameBox
    manuever would look like this::

        F F F
        F P F
        F F F

    where F stands for frame and P for picture. It is clear that in 2D, the split
    is into 8 frame pieces, and in 3D the equivalent would be 27-1=26 pieces.

    Parameters
    ----------
    frame_dimensions: Tuple[cm, cm, cm]
        The outer dimensions of the frame, in cm.
    picture_dimensions: Tuple[cm, cm, cm]
        The outer dimensions of the picture, in cm.
    frame_name: PurePath
        The name given to the Frame node. Must not contain '/'
    picture_name: PurePath
        The name given to the Picture node. Must not contain '/'
    frame_resolution: Tuple[cm, cm, cm]
        The resolution for the external frame. Defaults to minimal splits.
    picture_resolution: Tuple[cm, cm, cm]
        The resolution for the internal picture. Defaults to minimal splits.
    frame_mixture: Mixture
        The material the frame is made out of.
    picture_mixture: Mixture
        The material the picture is made out of.
    picture_translation: Tuple[cm, cm, cm]
        The translation of the picture center of mass from the frame center of mass.
    transform: Transform
        A transformation to apply to the entire FrameBox object.

    Returns
    -------
    Tree
        A (2-3)-level tree object. The top is the frame, on the second level there
        are both the internal frame nodes and the external picture node. The picture
        node could be 1-2 levels, depending on if it is split (and then it is a SplitBox)
        or if it isn't, in which case it is a trivial BoxTree.

    """
    picture_split = tuple(split(picture_dimensions, picture_resolution))
    picture_transform = Transform(translation=picture_translation)
    picture = (
        SplitBox(
            dimensions=picture_dimensions,
            mixture=picture_mixture,
            name=picture_name,
            resolution=picture_resolution,
            transform=picture_transform,
        )
        if any(v > 1 for v in picture_split)
        else BoxTree(
            dimensions=picture_dimensions, mixture=picture_mixture, name=picture_name, transform=picture_transform
        )
    )
    frame_dimensions = np.array(frame_dimensions)
    picture_dimensions = np.array(picture_dimensions)
    picture_translation = np.array(picture_translation)
    return _framebox(
        picture=picture,
        frame_dimensions=frame_dimensions,
        picture_dimensions=picture_dimensions,
        frame_name=frame_name,
        frame_resolution=frame_resolution,
        frame_mixture=frame_mixture,
        picture_translation=picture_translation,
        transform=transform,
    )


def ExcludeFrame(
    *,
    frame_dimensions: tuple[cm, cm, cm],
    picture_dimensions: tuple[cm, cm, cm],
    frame_name: PurePath,
    picture_name: PurePath,
    picture_resolution: tuple[cm, cm, cm] = (math.inf, math.inf, math.inf),
    frame_mixture: Mixture,
    picture_mixture: Mixture,
    picture_translation: tuple[cm, cm, cm] = (0.0, 0.0, 0.0),
    transform: Transform = identity,
) -> Tree:
    """Create a frame from one material containing an internal box (a picture,
    in the analogy this is framed as) made from another material.

    The picture can be set anywhere inside the frame and the picture can be
    split with a given resolution. The model is based on the picture being
    an exclusive child of the frame.

    This type of model has "hard to calculate volumes", though they are not that
    hard, since the volume is just the total volume minus the picture volume.

    Parameters
    ----------
    frame_dimensions: Tuple[cm, cm, cm]
        The outer dimensions of the frame, in cm.
    picture_dimensions: Tuple[cm, cm, cm]
        The outer dimensions of the picture, in cm.
    frame_name: PurePath
        The name given to the Frame node. Must not contain '/'
    picture_name: PurePath
        The name given to the Picture node. Must not contain '/'
    picture_resolution: Tuple[cm, cm, cm]
        The resolution for the internal picture. Defaults to minimal splits.
    frame_mixture: Mixture
        The material the frame is made out of.
    picture_mixture: Mixture
        The material the picture is made out of.
    picture_translation: Tuple[cm, cm, cm]
        The translation of the picture center of mass from the frame center of mass.
    transform: Transform
        A transformation to apply to the entire FrameBox object.

    """
    element = BoxTree(frame_dimensions, frame_mixture, frame_name, transform)
    picture_transform = Transform(translation=picture_translation)
    picture_split = tuple(split(picture_dimensions, picture_resolution))
    picture = (
        SplitBox(
            dimensions=picture_dimensions,
            mixture=picture_mixture,
            name=picture_name,
            resolution=picture_resolution,
            transform=picture_transform,
        )
        if any(v > 1 for v in picture_split)
        else BoxTree(
            dimensions=picture_dimensions, mixture=picture_mixture, name=picture_name, transform=picture_transform
        )
    )
    element.graft(picture, frame_name, ChildType.exclusive)
    return element


def excludeframe_to_framebox(
    excludeframe: Tree,
    picture_name: PurePath,
    frame_resolution: tuple[cm, cm, cm] = (math.inf, math.inf, math.inf),
) -> Tree:
    """Convert an existing ExcludeFrame tree structure to a FrameBox tree structure.

    ExcludeFrame structures don't split their frames into 3 conform mesh structures,
    so they are more efficient in terms of number of components. This makes them
    attractive in certain cases. However, we often need a conforming split that
    does not use exclusive progeny. So we need a converter.

    Parameters
    ----------
    excludeframe: Tree
        A Tree object created by the ExcludeFrame function.
        Basically, it's a nested pair of boxes where the inner is exclusive
        to the outer one. The inner box could be a tree of any shape, so long
        as its root has a Box geometry.
    picture_name: PurePath
        The path to the nested inner box.
    frame_resolution: (cm, cm, cm)
        The resolution to the split the frame into. In FrameBox objects we can
        split the frame at a higher resolution if we want to than just the
        conforming split.

    Returns
    -------
    A tree object that is equivalent to the one that would have been created
    by FrameBox if the original ExcludeFrame data was used in FrameBox directly.

    """
    frame_name, frame = list(excludeframe.roots())[0]
    frame_mixture = frame.mixture
    # Safe because we know the root geometry is a Box geometry
    frame_dimensions = frame.geometry.dimensions  # type:ignore
    picture_transform = excludeframe.get_transform(frame_name / picture_name)
    picture_translation = np.asarray(picture_transform.translation).reshape(3)
    picture = excludeframe.subtree(frame_name / picture_name)
    picture_root = list(picture.roots())[0][1]
    # Safe because we know the picture root geometry is a Box geometry
    picture_dimensions = picture_root.geometry.dimensions  # type:ignore
    return _framebox(
        picture=picture,
        frame_dimensions=frame_dimensions,
        picture_dimensions=picture_dimensions,
        frame_name=frame_name,
        frame_resolution=frame_resolution,
        frame_mixture=frame_mixture,
        picture_translation=picture_translation,
        transform=excludeframe.get_transform(frame_name),
    )


def SplitBox(
    dimensions: tuple[cm, cm, cm],
    mixture: Mixture,
    name: PurePath,
    resolution: tuple[cm, cm, cm],
    transform: Transform = identity,
) -> Tree:
    """Create a split box Tree object.

    Parameters
    ----------
    dimensions: (cm, cm, cm)
        The dimensions of the outer box, in cm.
    mixture: Mixture
        The material the box is made out of.
    name: PurePath
        The name to give the box. Should have no '/' in it.
    resolution: (cm, cm, cm)
        The resolution to split the box into in every dimension.
    transform: Transform
        The top level transform to apply to the split box.

    See Also
    --------
    :func:`BoxTree`

    Returns
    -------
    Tree
        A tree object with 2 levels. The root is the outer geometry and its
        inclusive progeny are the split box pieces.

    """
    element = Tree()
    element.nodes[name] = Node(Box(ORIGIN, dimensions), transform=transform)
    x, y, z = (
        np.linspace(-dimensions[i] / 2, +dimensions[i] / 2, num=splits + 1)
        for i, splits in enumerate(split(dimensions, resolution))
    )
    for dx, dy, dz in product(pairwise(x), pairwise(y), pairwise(z)):
        center = np.array((0.5 * (dx[0] + dx[1]), 0.5 * (dy[0] + dy[1]), 0.5 * (dz[0] + dz[1])))
        subdim = (dx[1] - dx[0], dy[1] - dy[0], dz[1] - dz[0])
        subname = PurePath(f"Piece:({', '.join(f'{v:.4e}' for v in center)})")
        newnode = Node(Box(ORIGIN, subdim), Transform(translation=center), mixture)
        element.nodes[name / subname] = newnode
    element.inclusive[name] = [(path, node) for path, node in element.nodes.items() if path != name]
    return element


def BoxTree(dimensions: tuple[cm, cm, cm], mixture: Mixture, name: PurePath, transform: Transform = identity) -> Tree:
    """Create a bare box shaped Tree object.

    Parameters
    -----------
    dimensions: (cm, cm, cm)
        The dimensions of the outer box, in cm.
    mixture: Mixture
        The material the box is made out of.
    name: PurePath
        The name to give the box. Should have no '/' in it.
    transform: Transform
        The top level transform to apply to the split box.

    Returns
    -------

    """
    element = Tree()
    element.nodes[name] = Node(Box(ORIGIN, dimensions), transform=transform, mixture=mixture)
    return element


PICTURE_PATH = PurePath("Picture")


def split_box_inside_tree(tree: Tree, box_path: PurePath, resolution: tuple[cm, cm, cm]):
    """
    Function that gets a tree and a path to a Node with box geometry inside the tree and splits the box,
    preserving the rest of the geometry. The change is done in place.
    The functions assume that the space inside the box is filled by the bounding box of the children
    of the node which is splitted.

    Parameters
    ----------
    tree: Tree
    box_path: PurePath
     The path of the spliited node.
    resolution: tuple[cm, cm, cm]
     The splitting resolution

    """
    node = tree.nodes[box_path]
    # noinspection PyTypeChecker
    box: Box = node.geometry
    if box_path.parent in tree.inclusive:
        parent_type = ChildType.inclusive
    elif box_path.parent in tree.exclusive and box_path in [p for p, n in tree.exclusive[box_path.parent]]:
        parent_type = ChildType.exclusive
    else:
        parent_type = ChildType.external_exclusive
    if box_path not in tree.exclusive | tree.inclusive | tree.external_exclusive:
        tree.cut(box_path)
        split_box = SplitBox(
            box.dimensions, node.mixture, PurePath(box_path.name), resolution, node.transform @ box.transform_
        )
        if box_path.parent == PurePath():
            tree.inclusive = split_box.inclusive
            tree.exclusive = split_box.exclusive
            tree.external_exclusive = split_box.external_exclusive
            tree.nodes = split_box.nodes
            return
        else:
            tree.graft(split_box, box_path.parent, parent_type)
            return
    if box_path in tree.inclusive or box_path in tree.external_exclusive:
        raise ValueError(f" The node at {box_path} must have only exclusive children")
    exclusive_nodes = tree.exclusive[box_path] if box_path in tree.exclusive else []
    exclusive_geometries = [n.geometry.transform(n.transform) for p, n in exclusive_nodes]
    holes_bbox = union_bounding_box(exclusive_geometries)
    translation = holes_bbox.center - box.center
    frame_box = FrameBox(
        frame_dimensions=box.dimensions,
        picture_dimensions=holes_bbox.dimensions,
        frame_name=PurePath(box_path.name),
        picture_name=PICTURE_PATH,
        frame_resolution=resolution,
        frame_mixture=node.mixture,
        picture_mixture=node.mixture,
        picture_translation=translation,
        transform=node.transform @ box.transform_,
    )
    subtrees = {p: tree.subtree(p) for p, n in exclusive_nodes}
    tree.cut(box_path)
    if box_path.parent == PurePath():
        tree.inclusive = frame_box.inclusive
        tree.exclusive = frame_box.exclusive
        tree.external_exclusive = frame_box.external_exclusive
        tree.nodes = frame_box.nodes
    else:
        tree.graft(frame_box, box_path.parent, parent_type)
    for path, sub_tree in subtrees.items():
        for p, root in sub_tree.roots():
            root.transform = identity
        tree.graft(sub_tree, box_path / PICTURE_PATH, ChildType.exclusive)
