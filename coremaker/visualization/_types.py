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
    half_width_x : float
        For SQUARE: half-width in x (dx/2).
        For HEXAGON: flat-to-flat half-pitch (pitch/2).
    half_width_y : float or None
        For SQUARE: half-width in y (dy/2).
        For HEXAGON: None (uniform pitch).

    """

    center_x: float
    center_y: float
    cell_shape: CellShape
    half_width_x: float
    half_width_y: float | None = None
