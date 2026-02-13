"""Utilities for elements.

"""
from pathlib import PurePath
from typing import Iterable, Generator

import numpy as np

from coremaker.protocols.geometry import Geometry
from coremaker.protocols.mixture import Mixture
from coremaker.transform import Transform, identity
from coremaker.tree import Tree, Node


def appropriate_resolution(dimensions: Iterable[float], split: Iterable[int]) \
        -> Generator[float, None, None]:
    """
    Return the appropriate resolution in each dimension to produce the splitting
    defined by the input split.

    Parameters
    ----------
    dimensions: Iterable[float]
        Dimensional length of a piece of geometry, in some length units.
    split: Iterable[int]
        The amount of pieces to have in each dimensions.

    Returns
    -------
    Dimensional resolution of a piece of geometry, in the same length units.
    """
    return (length * (1 / (cells_num - 1) + 1 / cells_num) / 2
            if cells_num > 1 else np.inf
            for length, cells_num in zip(dimensions, split))


def split(dimensions: Iterable[float], resolution: Iterable[float]
          ) -> Generator[int, None, None]:
    """Return the amount each dimension needs to be split in according to its
    resolution.

    Parameters
    ----------
    dimensions: Iterable[float]
        Dimensional length of a piece of geometry, in some length units.
    resolution: Iterable[float]
        Dimensional resolution of a piece of geometry, in the same length units.

    Returns
    -------
    The amount of pieces to have in each dimension.

    """
    return (int(d/r)+1 for d, r in zip(dimensions, resolution))


def symmetric_spacing(points: int,
                      step_vector: np.ndarray,
                      origin: np.ndarray = np.zeros(3)) -> list[np.ndarray]:
    # noinspection PyShadowingNames
    """Symmetrically positioned and evenly spaced locations around the origin.

    Evenly spaces points in 3D around some point along some line.
    
    Exampels
    ---------
    >>> from numpy import array
    >>> points = 5
    >>> step_vector = array((0, 2, 0))
    >>> origin = array((0, 0, 0))
    >>> symmetric_spacing(points, step_vector, origin)
    [array([ 0., -4.,  0.]), array([ 0., -2.,  0.]), array([0., 0., 0.]), array([0., 2., 0.]), array([0., 4., 0.])]
    """
    tile = np.tile(step_vector, (points, 1)).cumsum(axis=0)
    return list(tile - np.mean(tile, axis=0) + origin)


def single_node_tree(geometry: Geometry, mixture: Mixture, path: PurePath,
                     transform: Transform = identity) -> Tree:
    """
    function that creates a tree containing a single node.

    Parameters
    ----------
    geometry: Geometry
        The geoemtry of the node.
    mixture: Mixture
        The mixture of the node.
    path: PurePath
        The path of the node
    transform: Transform
        transform to be aplied to the node

    Returns
    -------
    Tree
     Tree that contains a single node.

    """
    tree = Tree()
    tree.nodes[path] = Node(geometry=geometry, mixture=mixture,
                            transform=transform)
    return tree
