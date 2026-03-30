"""Geometry-to-patch conversion for matplotlib rendering."""

import numpy as np

from coremaker.visualization.types import CellGeometry, CellShape

try:
    from matplotlib.patches import Polygon, Rectangle
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def make_patch(geom: CellGeometry, **kwargs) -> Rectangle | Polygon:
    """Create a matplotlib patch for a single cell.

    Parameters
    ----------
    geom : CellGeometry
        The cell geometry to render.
    **kwargs
        Additional keyword arguments passed to the matplotlib patch constructor.

    Returns
    -------
    Rectangle or Polygon

    """
    if geom.cell_shape == CellShape.SQUARE:
        w = geom.width_x
        h = geom.width_y if geom.width_y is not None else geom.width_x
        return Rectangle(
            (geom.center_x - w / 2, geom.center_y - h / 2),
            width=w,
            height=h,
            **kwargs,
        )
    elif geom.cell_shape == CellShape.HEXAGON:
        # Flat-topped hexagon: apothem = pitch/2
        # Circumradius from apothem: r = apothem / (sqrt(3)/2)
        apothem = geom.width_x / 2
        r = apothem / (np.sqrt(3) / 2)
        # Flat-topped orientation: first vertex at 0 degrees, rotated by 30 deg
        angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        vertices = np.column_stack(
            [
                geom.center_x + r * np.cos(angles),
                geom.center_y + r * np.sin(angles),
            ]
        )
        return Polygon(vertices, closed=True, **kwargs)
    else:
        raise ValueError(f"Unsupported cell shape: {geom.cell_shape}")
