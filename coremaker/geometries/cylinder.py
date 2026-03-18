"""Cylindrical finite geometries."""

from typing import Literal

import numpy as np
from ramp_core.serializable import Serializable
from scipy.linalg import norm as norm2

from coremaker.geometries.box import Box
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.util import DECIMAL_PRECISION, allclose, comma_format, isclose
from coremaker.transform import Transform
from coremaker.units import cm


def cylinder_bounding_box(center, axis, radius, length):
    axis = np.array(axis)
    axis = axis / norm2(axis, ord=2)
    dx = length * np.abs(axis) + 2 * radius * norm2(np.tile(axis, (3, 1)) - np.diag(axis), axis=1)
    return Box(tuple(center), tuple(dx))


class FiniteCylinder(Serializable):
    """Represents a finite cylinder in any direction"""

    ser_identifier = "FiniteCylinder"

    def __init__(self, center: tuple[cm, cm, cm], radius: cm, length: cm, axis: tuple[float, float, float]):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The location of the center compared to the origin in cm.
        radius: cm
            The radius of the cylinder, in cm.
        length: cm
            The axial length of the cylinder, in cm.
        axis: tuple[float, float, float]
            The symmetry axis of the cylinder.
        """
        self.center = tuple(float(v) for v in np.round(center, DECIMAL_PRECISION))
        self.radius = radius
        self.length = length
        if axis == (0.0, 0.0, 0.0):
            raise ValueError("Cylinder geometry direction cannot be 0")
        self.axis = axis

    def _canonical_axis(self) -> np.ndarray:
        axis = np.asarray(self.axis)
        return axis / norm2(axis, ord=2)

    @classmethod
    def cardinal(cls, center: tuple[cm, cm, cm], radius: float, length: float, axis: Literal["x", "y", "z", 0, 1, 2]):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The location of the center compared to the origin in cm.
        radius: cm
            The radius of the cylinder, in cm.
        length: cm
            The axial length of the cylinder, in cm.
        axis: Literal[0,1,2,'x','y','z']
            The symmetry axis of the cylinder.
            'x' and 0 are shortcuts for the (1,0,0) axis.
            'y' and 1 are shortcuts for the (0,1,0) axis.
            'z' and 2 are shortcuts for the (0,0,1) axis.
        """
        return cls(
            center,
            radius,
            length,
            {
                0: (1.0, 0.0, 0.0),
                1: (0.0, 1.0, 0.0),
                2: (0.0, 0.0, 1.0),
                "x": (1.0, 0.0, 0.0),
                "y": (0.0, 1.0, 0.0),
                "z": (0.0, 0.0, 1.0),
            }[axis],
        )

    @property
    def volume(self) -> float:
        return self.length * np.pi * self.radius**2

    @property
    def surfaces(self) -> tuple[Plane, Plane, Cylinder]:
        """The surfaces that make up this finite geometry."""
        axis = self._canonical_axis()
        center = np.array(self.center)
        a1, a2, a3 = axis
        bp = axis @ center + self.length / 2
        bm = axis @ center - self.length / 2
        return (
            Plane(a1, a2, a3, bm),
            -Plane(a1, a2, a3, bp),
            Cylinder(self.center, self.radius, self.axis, inside=True),
        )

    def transform(self, transform: Transform) -> "FiniteCylinder":
        """Allows for translation and rotation of this geometry.

        Parameters
        ----------
        transform: Transform
            The Transform to change this by.

        Returns
        -------
        A new FiniteCylinder.

        """
        return FiniteCylinder(
            tuple(transform @ np.asarray(self.center)),
            self.radius,
            self.length,
            tuple(transform.rotmat @ np.asarray(self.axis)),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, FiniteCylinder):
            return NotImplemented
        axis, oaxis = self._canonical_axis(), other._canonical_axis()
        return (
            allclose(self.center, other.center)
            and allclose(abs(axis @ oaxis), 1)
            and all(isclose(getattr(self, d), getattr(other, d)) for d in ("radius", "length"))
        )

    def __hash__(self):
        return hash((self.center, self.radius, self.length, tuple(self._canonical_axis())))

    def __repr__(self) -> str:
        return (
            f"FiniteCylinder<Center: {comma_format(self.center)}, "
            f"Radius: {self.radius:.3e}, Length: {self.length:.3e}, "
            f"Axis: {comma_format(self.axis)}>"
        )

    def bounding_box(self):
        return cylinder_bounding_box(self.center, self.axis, self.radius, self.length)
