"""Core data types for the visualization library."""

from dataclasses import dataclass
from enum import Enum


class CellShape(Enum):
    """Shape of individual assembly cells in the core grid."""

    SQUARE = "square"
    HEXAGON = "hexagon"


@dataclass(frozen=True)
class CellGeometry:
    """Renderable geometry of one assembly site.

    Parameters
    ----------
    center_x : float
        Absolute x position in cm.
    center_y : float
        Absolute y position in cm.
    cell_shape : CellShape
        Shape of this cell.
    width_x : float
        Full edge-to-edge extent in x.
        For SQUARE: cell width (dx).  For HEXAGON: flat-to-flat pitch.
    width_y : float or None
        Full edge-to-edge extent in y.
        For SQUARE: cell height (dy).  For HEXAGON: None (uniform pitch).

    """

    center_x: float
    center_y: float
    cell_shape: CellShape
    width_x: float
    width_y: float | None = None
