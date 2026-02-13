"""
file that contains tools to intersect 3d objects with a plane parallel to the xy plane to get 2d objects.
This is useful when a simplification of the geometry is desired, for example when modeling unit cells for
cross-section generation. This can also be used for 2d-1d solvers.
"""

import numpy as np
from multipledispatch import dispatch

from coremaker.core import Core
from coremaker.geometries.annulus import Annulus, Ring
from coremaker.geometries.ball import Ball, Circle
from coremaker.geometries.box import Box, Rectangle
from coremaker.geometries.cylinder import FiniteCylinder
from coremaker.geometries.hex import HexPrism, Hexagon
from coremaker.geometries.holed import ConcreteHoledGeometry
from coremaker.geometries.infinite import _InfiniteGeometry
from coremaker.geometries.union import ConcreteUnionGeometry
from coremaker.grid import CartesianGrid, SpacedGrid
from coremaker.protocols.grid import Lattice, Grid
from coremaker.transform import Transform
from coremaker.transform import identity
from coremaker.tree import Tree, Node


@dispatch(Box, object)
def intersect_geometry(geo: Box, z0: float) -> Rectangle | None:
    minv, maxv = (geo.center[2] - geo.dimensions[2] / 2,
                  geo.center[2] + geo.dimensions[2] / 2)
    if not minv < z0 < maxv:
        return None
    return Rectangle(center=np.zeros(2), dimensions=geo.dimensions[:2],
                     transform=restrict_transform(geo.transform_))


@dispatch(ConcreteUnionGeometry, object)
def intersect_geometry(geo: ConcreteUnionGeometry, z0: float) -> ConcreteUnionGeometry:
    geometries_2d = [x for g in geo.geometries if (x := intersect_geometry(g, z0))]
    return ConcreteUnionGeometry(geometries_2d)


@dispatch(Ball, object)
def intersect_geometry(geo: Ball, z0: float) -> Circle | None:
    if np.abs(geo.center[2] - z0) > geo.radius:
        return None
    r = np.sqrt(geo.radius ** 2 - (z0 - geo.center[2]) ** 2)
    return Circle(geo.center[:2], r)


@dispatch(HexPrism, object)
def intersect_geometry(geo: HexPrism, z0: float) -> Hexagon | None:
    if not geo.center[2] - geo.height / 2 < z0 < geo.center[2] + geo.height / 2:
        return None
    return Hexagon(geo.center[:2], geo.pitch)


@dispatch(_InfiniteGeometry, object)
def intersect_geometry(geo: _InfiniteGeometry, z0: float) -> _InfiniteGeometry:
    return geo


@dispatch(ConcreteHoledGeometry, object)
def intersect_geometry(geo: ConcreteHoledGeometry, z0: float) -> ConcreteHoledGeometry | None:
    geo_2d = intersect_geometry(geo.inclusive, z0)
    if not geo_2d:
        return None
    internal_exclusive_2d = [hole_2d for hole in geo.internal_exclusives
                             if (hole_2d := intersect_geometry(hole, z0))]
    external_exclusive_2d = [hole_2d for hole in geo.external_exclusives
                             if (hole_2d := intersect_geometry(hole, z0))]
    return ConcreteHoledGeometry(geo_2d,
                                 internal_exclusive_2d,
                                 external_exclusive_2d)


@dispatch(FiniteCylinder, object)
def intersect_geometry(geo: FiniteCylinder, z0: float) -> Circle | None:
    if not np.allclose(geo.axis[:-1], np.zeros(2)):
        if np.abs(z0 - geo.center[2]) > geo.radius:
            return None
        raise NotImplementedError(
            "Intersecting cylinders not perpendicular to the plane is not supported yet")
    if geo.center[2] - geo.length / 2 < z0 < geo.center[2] + geo.length / 2:
        return Circle(geo.center[:2], geo.radius)
    return None


@dispatch(Annulus, object)
def intersect_geometry(geo: Annulus, z0: float) -> Ring | None:
    if not np.allclose(geo.axis[:-1], np.zeros(2)):
        if np.abs(z0 - geo.center[2]) > geo.outer_radius:
            return None
        raise NotImplementedError(
            "Intersecting cylinders not perpendicular to the plane is not supported yet")
    if geo.center[2] - geo.length / 2 < z0 < geo.center[2] + geo.length / 2:
        return Ring(geo.center[:2], geo.inner_radius, geo.outer_radius)
    return None


@dispatch(object, object)
def intersect_geometry(g: object, z0: float):
    raise NotImplementedError("Geometry intersection isn't available for geometries of "
                              f"type {type(g)}. Tried to intersect {g}")


@dispatch(object, object)
def intersect_geometry(g: object, z0: float):
    raise NotImplementedError("Geometry intersection isn't available for geometries of "
                              f"type {type(g)}. Tried to intersect {g}")


def intersect_tree(element: Tree, z0: float) -> Tree:
    """
    Function to intersect a 3d Tree with a plane and obtain a 2d Tree

    Parameters
    ----------
    element: Tree
     The 3d Tree
    z0: float
     The height of the plane used for intersection

    Returns
    -------
    Tree
     The 2d tree
    """
    memdict = {}
    result = Tree()
    for path, node in element.nodes.items():
        if dim2geometry := intersect_geometry(node.geometry.transform(element.get_transform(path, memdict)),
                                              z0):
            if isinstance(node, Lattice):
                result.nodes[path] = node
            else:
                result.nodes[path] = Node(dim2geometry, identity,
                                          mixture=node.mixture)
    for element_edges, result_edges in zip(
            [element.inclusive, element.exclusive, element.external_exclusive],
            [result.inclusive, result.exclusive, result.external_exclusive]):
        for path, nodes in element_edges.items():
            if path in result.nodes:
                result_edges[path] = [(node[0], result[node[0]]) for node in nodes if
                                      node[0] in result.nodes]
    return result


@dispatch(CartesianGrid, object)
def intersect_grid(grid: CartesianGrid, z0: float) -> CartesianGrid:
    """
    function to intersect a cartesian grid. The function intersects all the lattices and all the rods
    of the grid.

    Parameters
    ----------
    grid: CartesianGrid
     The grid.
    z0: float
     The intersection height

    Returns
    -------
    CartesianGrid
    """
    return CartesianGrid(grid.lattice.origin, grid.lattice.shape, grid.lattice.dimensions, None,
                         grid.lattice.mixture,
                         {site: intersect_tree(rod, z0) for site, rod in grid.contents.items()})


@dispatch(SpacedGrid, object)
def intersect_grid(grid: SpacedGrid, z0: float) -> SpacedGrid:
    """
    function to intersect a spaced grid. The function intersects all the lattices and all the rods
    of the grid.

    Parameters
    ----------
    grid: SpacedGrid
     The grid.
    z0: float
     The intersection height

    Returns
    -------
    SpacedGrid
    """
    center = sum(l.origin / 4 for l in grid.lattices)
    return SpacedGrid(center, (grid._shape[0] * 2, grid._shape[1] * 2), grid.lattices[0].dimensions, None,
                      grid.space_dx, grid.space_dy, grid.lattices[0].mixture,
                      {site: intersect_tree(rod, z0) for site, rod in grid.contents.items()})


@dispatch(object, object)
def intersect_grid(grid: Grid, z0: float):
    """
    function to intersect a grid. The function intersects all the lattices and all the rods
    of the grid.
    """
    raise NotImplementedError("Grid intersection isn't available for grids of "
                              f"type {type(grid)}. Tried to intersect {grid}")


def restrict_transform(transform: Transform) -> Transform:
    """
    returns the transform restricted to 2d

    Parameters
    ----------
    transform: Transform
     transform that should preserve the z axis

    Returns
    -------
    Transform
     The transform restricted to 2d
    """
    if not np.allclose(transform.rotation.as_matrix() @ np.array([0, 0, 1]), np.array([0, 0, 1])):
        raise ValueError("Can't interesect a box non parralel to the x,y,z axi")
    translation = transform.translation.flatten()
    translation[-1] = 0
    rotation_matrix = transform.rotation.as_matrix()
    rotation_matrix[-1, :] = np.array([0, 0, 1])
    rotation_matrix[:, -1] = np.array([0, 0, 1])
    return Transform(translation=translation, rotation=rotation_matrix)


def intersect_core(core: Core, z0: float) -> Core:
    """
    Function used to intersect a 3d Core with a plane to obtain a 3d Core

    Parameters
    ----------
    core:Core
     The 3d core
    z0:float
     The height of the plane used for intersection

    Returns
    -------
    Core
     The 2d core
    """
    grid_2d = intersect_grid(core.grid, z0)
    core_tree = intersect_tree(core.tree, z0)
    for path, node in core_tree.nodes.items():
        if isinstance(node, Lattice):
            for lattice in grid_2d.lattices:
                if np.all(lattice.origin == node.origin):
                    core_tree.nodes[path] = lattice
    return Core(grid_2d, core.aliases, core_tree,
                intersect_geometry(core.outer_geometry, z0))
