import math
from itertools import pairwise, product, repeat
from pathlib import PurePath
from typing import Sequence

import numpy as np
from cytoolz import first

from coremaker.elements.box import ORIGIN
from coremaker.elements.util import appropriate_resolution, split
from coremaker.geometries import Annulus, FiniteCylinder
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.mixture import Mixture
from coremaker.transform import Transform
from coremaker.tree import Node, Tree
from coremaker.units import cm


def _cylindrically_symmetric_geometry(
    radius: cm, length: cm, axis: tuple[float, float, float], center: tuple[cm, cm, cm] = ORIGIN, inner_radius: cm = 0.0
) -> Geometry:
    """
    Returns the geometry of the cylinder/annulus specified by the input
    parameters. A cylinder here is considered an annulus with inner_radius
    of 0.
    Parameters
    ----------
    radius: tuple[cm, cm, cm]
        The outer radius of the cylinder/annulus.
    length: cm
        The length of the cylinder/annulus.
    axis: tuple[float, float, float]
        The symmetry axis of the cylinder/annulus.
    center - tuple[cm, cm, cm]
        The location of the center compared to the origin in cm.
    inner_radius: cm
        The inner radius of the annulus. 0. if the requested geometry is
        a cylinder.
    """
    if inner_radius == 0.0:
        return FiniteCylinder(center, radius, length, axis)
    else:
        return Annulus(center, inner_radius, radius, length, axis)


# noinspection PyPep8Naming
def AnnulusTree(
    inner_radius: cm,
    outer_radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    center: tuple[cm, cm, cm] = ORIGIN,
):
    """
    Returns a tree that models an element in the shape of an annulus.
    Parameters
    ----------
    inner_radius - The inner radius of the annulus.
    outer_radius - The outer radius of the annulus.
    length - The length of the annulus.
    mixture - The mixture of the element.
    root_path - The path of the root node of the tree object.
    axis - The symmetry axis of the annulus.
    center - The location of the center compared to the origin in cm.
    """
    element = Tree()
    geo = _cylindrically_symmetric_geometry(
        radius=outer_radius, length=length, axis=axis, center=center, inner_radius=inner_radius
    )
    element.nodes[root_path] = Node(geo, mixture=mixture)
    return element


# noinspection PyPep8Naming
def CylinderTree(
    radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    center: tuple[cm, cm, cm] = ORIGIN,
):
    """
    Returns a tree that models an element in the shape of a cylinder.
    Parameters
    ----------
    radius - The outer radius of the cylinder.
    length - The length of the cylinder.
    mixture - The mixture of the element.
    root_path - The path of the root node of the tree object.
    axis - The symmetry axis of the cylinder.
    center - The location of the center compared to the origin in cm.
    """
    return AnnulusTree(
        outer_radius=radius,
        length=length,
        mixture=mixture,
        root_path=root_path,
        axis=axis,
        center=center,
        inner_radius=0.0,
    )


def _splits_num(radius: cm, resolution: cm, inner_radius: cm = 0.0) -> int:
    """
    Return the minimal integer number of splits necessary to split a cylinder
    or annulus radially according to the specified resolution.
    Here, it is assumed the radial pieces are of equal volume.

    Parameters
    ----------
    radius - The outer radius.
    resolution - The maximal allowed distance between successive radii in the
        radially split cylinder/annulus.
    inner_radius - The inner radius of the cylinder/annulus.

    Returns
    -------
    The number of splits in the radial direction.
    """
    if resolution == math.inf:
        return 1
    elif np.isclose(resolution, radius - inner_radius, rtol=1e-10):
        return 2
    else:
        return int(((radius**2) - (inner_radius**2)) / (2 * resolution * inner_radius + resolution**2)) + 1


def radial_split(radius: cm, resolution: cm = math.inf, inner_radius: cm = 0.0) -> list[cm]:
    """
    Return an iterable of all the radial bounds of the radial pieces of
    the chunked cylindrically symmetric structure.
    The splitting is performed so that each piece will be of equal volume.

    Parameters
    ----------
    radius - The outer radius.
    resolution - The maximal allowed distance between successive radii in the
        radially split cylinder/annulus.
    inner_radius - The inner radius of the cylinder/annulus (0. in the case
    of a cylinder).

    Returns
    -------
    The number of splits in the radial direction.
    """
    n = _splits_num(radius, resolution, inner_radius)

    return [np.sqrt((i / n * (radius**2)) - (i / n - 1) * inner_radius**2) for i in range(n + 1)]


def axial_split(length: cm, resolution: cm = math.inf) -> list[cm]:
    """
    Returns the axial split of the cylinder/annulus at the given resolution.
    Specifically, the return value is a list where each entry is the size of
    a corresponding axial slice of the cylinder/annulus.

    Parameters
    ----------
    length - The total length of the cylinder/annulus
    resolution - The resolution of the splitting.
    """
    (n,) = split([length], [resolution])
    return list(repeat(length / n, n))


def _piece_centers(lengths: Sequence[cm], axis: tuple[float, float, float], center: tuple[cm, cm, cm] = ORIGIN):
    """
    Returns the center of each of the axial slices of the cylinder/annulus.

    Parameters
    ----------
    lengths: Sequence[cm]
        The lengths of each axial slice
    axis: tuple[float, float, float]
        The symmetry axis direction vector
    center: tuple[cm, cm, cm]
        The cylinder/annulus center location in cm.

    """
    return [tuple(np.asarray(v) + center) for v in _piece_translations(lengths, axis)]


def _piece_translations(lengths: Sequence[cm], axis: tuple[float, float, float]):
    """Returns transform of each of the axial slices of the cylinder/annulus.

    Parameters
    ----------
    lengths: Sequence[cm]
        The lengths of each axial slice
    axis: tuple[float, float, float]
        The symmetry axis direction vector

    """
    axis = np.asarray(axis) / np.linalg.norm(axis)
    lengths = [0.0] + list(lengths)
    steps_size = np.array([(x + y) / 2 for x, y in pairwise(lengths)])
    steps = np.outer(steps_size, axis)
    length = np.sum(lengths)
    return [tuple(step - length / 2 * axis) for step in steps.cumsum(axis=0)]


# noinspection PyPep8Naming
def ChunkedAnnulusTree(
    inner_radius: cm,
    outer_radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    resolution: tuple[cm, cm] = (math.inf, math.inf),
    center: tuple[cm, cm, cm] = ORIGIN,
) -> Tree:
    """
    Returns a tree object that models an element with the outer geometry of
    an annulus, but is chunked into many pieces of equal volume.

    Parameters
    ----------
    inner_radius:
        The inner radius of the annulus.
    outer_radius:
        The outer radius of the annulus.
    length:
        The length of the annulus.
    mixture:
        The mixture of the element.
    root_path:
        The path to the root node of the tree object.
    axis:
        The axis of symmetry of the annulus.
    resolution: tuple[cm, cm]
        The radial resolution and axial resolution of the annulus, respectively.
        The resolution defines the maximal size of a piece in the corresponding
        direction.
    center:
        The location of the center compared to the origin in cm.
    """
    element = Tree()
    outer_geo = _cylindrically_symmetric_geometry(outer_radius, length, axis, center, inner_radius)
    element.nodes[root_path] = Node(outer_geo)
    radial_resolution, axial_resolution = resolution
    radii = radial_split(outer_radius, radial_resolution, inner_radius=inner_radius)
    lengths = axial_split(length, axial_resolution)
    transforms = _piece_translations(lengths, axis)
    piece_paths = []
    for (transform, piece_length), (inner_radius, outer_radius) in product(zip(transforms, lengths), pairwise(radii)):
        geometry = _cylindrically_symmetric_geometry(outer_radius, piece_length, axis, center, inner_radius)
        name = f"Piece:({inner_radius:.1e}<r<{outer_radius:.1e}, c={', '.join(f'{c + t:.1e}' for c, t in zip(center, transform))})"
        path = root_path / PurePath(name)
        element.nodes[path] = Node(geometry, mixture=mixture, transform=Transform(transform))
        piece_paths.append(path)
    element.inclusive[root_path] = [(path, element.nodes[path]) for path in piece_paths]
    return element


# noinspection PyPep8Naming
def ChunkedCylinderTree(
    radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    resolution: tuple[cm, cm] = (math.inf, math.inf),
    center: tuple[cm, cm, cm] = ORIGIN,
) -> Tree:
    """
    Returns a tree object that models an element with the outer geometry of
    a cylinder, but is chunked into many pieces of equal volume.

    Parameters
    ----------
    radius:
        The radius of the cylinder.
    length:
        The length of the cylinder.
    mixture:
        The mixture of the cylinder.
    root_path:
        The path to the root node of the tree object.
    axis:
        The axis of symmetry of the cylinder.
    resolution: tuple[cm, cm]
        The radial resolution and axial resolution of the cylinder, respectively.
        The resolution defines the maximal size of a piece in the corresponding
            direction.
    center:
        The location of the center compared to the origin in cm.
    """
    return ChunkedAnnulusTree(
        outer_radius=radius,
        length=length,
        mixture=mixture,
        root_path=root_path,
        axis=axis,
        resolution=resolution,
        center=center,
        inner_radius=0.0,
    )


def UnequallyChunkedAnnulusTree(
    inner_radius: cm,
    outer_radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    lengths: Sequence[cm],
    center: tuple[cm, cm, cm] = ORIGIN,
) -> Tree:
    """
    Returns a tree object that models an element with the outer geometry of
    an annulus, but is chunked into many pieces of unequal volume.

    Parameters
    ----------
    radius:
        The radius of the cylinder.
    length:
        The length of the cylinder.
    mixture:
        The mixture of the cylinder.
    root_path:
        The path to the root node of the tree object.
    axis:
        The axis of symmetry of the cylinder.
    lengths:
        The split to different lengths along the cylinder axis.
    center:
        The location of the center compared to the origin in cm.
    """
    element = Tree()
    outer_geo = _cylindrically_symmetric_geometry(outer_radius, length, axis, center, inner_radius)
    element.nodes[root_path] = Node(outer_geo)
    radii = radial_split(outer_radius, inner_radius=inner_radius)
    transforms = _piece_translations(lengths, axis)
    piece_paths = []
    for (transform, piece_length), (inner_radius, outer_radius) in product(zip(transforms, lengths), pairwise(radii)):
        geometry = _cylindrically_symmetric_geometry(outer_radius, piece_length, axis, center, inner_radius)
        name = f"Piece:({inner_radius:.1e}<r<{outer_radius:.1e}, c={', '.join(f'{c + t:.1e}' for c, t in zip(center, transform))})"
        path = root_path / PurePath(name)
        element.nodes[path] = Node(geometry, mixture=mixture, transform=Transform(transform))
        piece_paths.append(path)

    element.inclusive[root_path] = [(path, element.nodes[path]) for path in piece_paths]
    return element


def UnequallyChunkedCylinderTree(
    radius: cm,
    length: cm,
    mixture: Mixture,
    root_path: PurePath,
    axis: tuple[float, float, float],
    lengths: Sequence[cm],
    center: tuple[cm, cm, cm] = ORIGIN,
) -> Tree:
    """
    Returns a tree object that models an element with the outer geometry of
    a cylinder, but is chunked into many pieces of unequal volume.

    Parameters
    ----------
    radius:
        The radius of the cylinder.
    length:
        The length of the cylinder.
    mixture:
        The mixture of the cylinder.
    root_path:
        The path to the root node of the tree object.
    axis:
        The axis of symmetry of the cylinder.
    lengths:
        The split to different lengths along the cylinder axis.
    center:
        The location of the center compared to the origin in cm.
    """
    return UnequallyChunkedAnnulusTree(
        outer_radius=radius,
        length=length,
        mixture=mixture,
        root_path=root_path,
        axis=axis,
        lengths=lengths,
        center=center,
        inner_radius=0.0,
    )


def _f(r: float, o_r: float, i: int) -> float:
    """Equal volume split"""
    return math.sqrt(((o_r**2 - r**2) / i) + r**2) - r


def appropriate_radial_resolution(radial_split_num: int, outer_radius: cm, inner_radius: cm = 0.0) -> cm:
    """A tool to determine the correct radial resolution for a desired
    amount of radial pieces.

    Parameters
    ----------
    radial_split: int
        The amount of radial pieces.
    outer_radius: cm
        The outer radius of the annulus/cylinder
    inner_radius: cm
        The inner radius of the annulus, 0. for a cylinder.

    Returns
    -------
    cm
        A resolution that corresponds to the desired amount of radial pieces.

    Examples
    --------
    >>> n, outer_r, inner_r = 10, 10., 0.
    >>> resolution = appropriate_radial_resolution(n, outer_r, inner_r)
    >>> resolution
    3.2478054967508565
    >>> radial_regions_num = len(radial_split(outer_r, resolution, inner_r)) - 1
    >>> radial_regions_num
    10

    """
    if radial_split_num == 1:
        return math.inf
    return (_f(inner_radius, outer_radius, radial_split_num) + _f(inner_radius, outer_radius, radial_split_num - 1)) / 2


def appropriate_axial_resolution(axial_split: int, length: cm) -> cm:
    """
    A tool to determine the correct axial resolution for a desired amount of
    axial pieces.

    Parameters
    ----------
    axial_split: int
        The amount of axial pieces.
    length: cm
        The length of the cylinder/annulus.

    Returns
    -------
    cm
        A resolution that corresponds to the desired amount of axial pieces.
    """
    return first(appropriate_resolution((length,), (axial_split,)))
