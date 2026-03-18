"""A cylindrical surface object."""

import math

import numpy as np
from ramp_core.serializable import Serializable
from scipy.linalg import norm as norm2

from coremaker.protocols.surface import Surface
from coremaker.surfaces.util import comma_format
from coremaker.transform import Transform
from coremaker.units import cm


class Cylinder(Serializable):
    """A cylinder.

    For most methods' explanation see the :class:`~coremaker.protocols.surface.Surface`
    protocol.

    Parameters
    ----------
    center: tuple[cm, cm, cm]
        The location of the center of the cylinder in cm, compared to the origin.
    radius: cm
        The radius of the cylinder, in cm.
    axis: tuple[float,float,float]
        Axis orientation. This should be a unit vector, but non-unit vectors are
        supported as well.
    inside: bool
        A flag for whether we are discussing the inside or outside the cylinder.
    """

    ser_identifier = "SurfCyl"
    __slots__ = ["center", "radius", "axis", "inside"]

    def __init__(self, center: tuple[cm, cm, cm], radius: cm, axis: tuple[float, float, float], *, inside: bool):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The location of the center compared to the origin in cm.
        radius: cm
            The radius of the cylinder, in cm.
        axis: (cm, cm, cm)
            Axis orientation.
        inside: bool
            A flag for whether we are discussing the inside or outside the cylinder.
        """
        if axis == (0.0, 0.0, 0.0):
            raise ValueError("Cylinder axis cannot be 0.")
        self.axis = axis
        self.center = center
        self.radius = radius
        self.inside = inside

    def __hash__(self):
        return hash((self.center, self.radius, frozenset((self.axis, tuple(-x for x in self.axis))), self.inside))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cylinder):
            return NotImplemented
        return all(np.all(getattr(self, attr) == getattr(other, attr)) for attr in ("center", "radius", "inside")) and (
            np.all(np.array(self.axis) == np.array(other.axis)) or np.all(np.array(self.axis) == -np.array(other.axis))
        )

    def __neg__(self) -> "Cylinder":
        return Cylinder(center=self.center, radius=self.radius, axis=self.axis, inside=not self.inside)

    def transform(self, transform: Transform) -> "Cylinder":
        return Cylinder(
            tuple(transform @ np.array(self.center)),
            self.radius,
            axis=tuple(transform.rotmat @ self.axis),
            inside=self.inside,
        )

    transform.__doc__ = Surface.transform.__doc__

    def normal(self, point: tuple[cm, cm, cm]) -> tuple[float, float, float]:
        vec = np.array(point) - np.array(self.center)
        axis = np.array(self.axis)
        vec = vec - (vec @ axis) * axis
        sign = 1 if self.inside else -1
        vec = sign * vec / norm2(vec, ord=2)
        return tuple(vec)

    normal.__doc__ = Surface.normal.__doc__

    def isclose(self, other: "Cylinder") -> bool:
        """Tests whether two cylinders are considered "close".

        Parameters
        ----------
        other: Cylinder
            The other cylinder to compare with.

        """
        if not isinstance(other, Cylinder):
            return False
        axis1, axis2 = np.array(self.axis), np.array(other.axis)
        axis1, axis2 = axis1 / norm2(axis1, ord=2), axis2 / norm2(axis2, ord=2)
        center1, center2 = np.array(self.center), np.array(other.center)
        center1 = center1 - (center1 @ axis1) * axis1
        center2 = center2 - (center2 @ axis2) * axis2
        return (
            (self.inside == other.inside)
            and all(math.isclose(x, y, rel_tol=0, abs_tol=1e-5) for x, y in zip(center1, center2))
            and (
                all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(axis1, axis2))
                or all(math.isclose(x, -y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(axis1, axis2))
            )
            and math.isclose(self.radius, other.radius, rel_tol=0, abs_tol=1e-5)
        )

    def __repr__(self) -> str:
        return (
            f"Cylinder(center={comma_format(self.center)}, "
            f"radius={self.radius:.3e}, axis={comma_format(self.axis)}, "
            f"inside={self.inside}>"
        )
