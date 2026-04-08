"""Extract renderable geometry from a coreMaker Core object."""

import numpy as np

from coremaker.grids.cartgrid import CartesianGrid
from coremaker.grids.cartspaced import GeneralSpacedGrid, SpacedGrid

from coremaker.visualization.types import CellGeometry, CellShape


def cell_shape_for_grid(grid) -> CellShape:
    """Determine the CellShape for a grid type."""
    if isinstance(grid, (CartesianGrid, SpacedGrid, GeneralSpacedGrid)):
        return CellShape.SQUARE
    raise TypeError(f"Unsupported grid type: {type(grid)}")


def cell_dimensions_for_site(grid, site) -> tuple[float, float | None]:
    """Return (width_x, width_y) for a site.

    For CartesianGrid the dimensions are uniform across all sites.
    For SpacedGrid/GeneralSpacedGrid, each site may belong to a different
    lattice with potentially different dimensions.
    """
    if isinstance(grid, CartesianGrid):
        dims = grid.lattice.dimensions
        return float(dims[0]), float(dims[1])

    if isinstance(grid, (SpacedGrid, GeneralSpacedGrid)):
        lattice, _ = grid.site_index(site)
        dims = lattice.dimensions
        return float(dims[0]), float(dims[1])

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def site_geometry(core, site) -> CellGeometry:
    """Build a CellGeometry for one site from a Core object."""
    grid = core.grid
    transform = core.site_transform(site)
    translation = transform.translation.flatten()
    tx, ty = float(translation[0]), float(translation[1])

    shape = cell_shape_for_grid(grid)
    hw_x, hw_y = cell_dimensions_for_site(grid, site)

    return CellGeometry(
        center_x=tx,
        center_y=ty,
        cell_shape=shape,
        width_x=hw_x,
        width_y=hw_y,
    )


def all_site_geometries(core) -> dict[str, CellGeometry]:
    """Dict of site label to CellGeometry for every site in the core."""
    return {site: site_geometry(core, site) for site in core.grid.sites()}


def occupied_sites(core) -> set[str]:
    """Return the set of occupied site labels."""
    return set(core.grid.keys())


def site_labels_from_core(core) -> dict[str, str]:
    """Extract display labels from occupied elements (uses element.name if available)."""
    labels = {}
    grid = core.grid
    for site in grid.keys():
        element = grid[site]
        if hasattr(element, "name") and element.name is not None:
            labels[site] = element.name
    return labels


def site_rotations(core) -> dict[str, float]:
    """Return the z-rotation angle in degrees for each occupied site.

    The rotation is extracted from the root node transform of each rod element.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.

    Returns
    -------
    dict[str, float]
        Mapping of site label to rotation angle in degrees, normalised to [0, 360).

    """
    rotations: dict[str, float] = {}
    grid = core.grid
    for site in grid.keys():
        element = grid[site]
        _, root_node = next(element.roots())
        angle = float(np.degrees(root_node.transform.rotation.as_euler("xyz")[2]))
        rotations[site] = angle % 360
    return rotations
