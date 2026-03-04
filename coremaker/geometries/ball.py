"""A geometry representation of a ball.

"""

import numpy as np
from ramp_core.serializable import Serializable

from coremaker.geometries.box import Box
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.sphere import Sphere
from coremaker.surfaces.util import DECIMAL_PRECISION, allclose, comma_format, isclose
from coremaker.transform import Transform
from coremaker.units import cm


class Ball(Serializable):
    """A ball shape.
    """

    ser_identifier = "Ball"

    def __init__(self, center: tuple[cm, cm, cm] | np.ndarray, radius: cm):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The center compared to the origin, in cm.
        radius: cm
            The radius of the ball, in cm.
        """
        self.center = np.round(center, DECIMAL_PRECISION)
        self.radius = radius

    @property
    def surfaces(self) -> tuple[Sphere]:
        return Sphere(self.center, self.radius, inside=True),

    @property
    def volume(self) -> float:
        return 4 * np.pi / 3 * self.radius ** 3

    def __eq__(self, other) -> bool:
        if isinstance(other, Ball):
            return (allclose(self.center, other.center) and
                    isclose(self.radius, other.radius))
        return NotImplemented

    def __hash__(self):
        return hash((tuple(self.center), self.radius))

    def __repr__(self) -> str:
        return f"Ball<Center: {comma_format(self.center)}, Radius: {self.radius:.3e}>"

    def transform(self, transform: Transform) -> "Ball":
        return Ball(transform @ self.center, self.radius)

    def bounding_box(self):
        dimensions = np.full((3,), fill_value=2 * self.radius)
        return Box(tuple(self.center), tuple(dimensions))


class Circle(Serializable):
    """
    A 2d circle shape.
    """

    ser_identifier = "Circle"

    def __init__(self, center: tuple[cm, cm], radius: cm):
        """

        Parameters
        ----------
        center: tuple of length 2
            The center compared to the origin, in cm.
        radius: cm
            The radius of the ball, in cm.
        """
        self.center = np.round(center, DECIMAL_PRECISION)
        self.radius = radius

    @property
    def surfaces(self) -> tuple[Cylinder]:
        return Cylinder(tuple(np.hstack([self.center, [0]])), self.radius, (0, 0, 1), inside=True),

    @property
    def volume(self) -> float:
        return np.pi * self.radius ** 2

    def __eq__(self, other) -> bool:
        if isinstance(other, Circle):
            return (allclose(self.center, other.center) and
                    isclose(self.radius, other.radius))
        return NotImplemented

    __hash__ = Ball.__hash__

    def __repr__(self) -> str:
        return f"Circle<Center: {comma_format(self.center)}, Radius: {self.radius:.3e}>"

    def transform(self, transform: Transform) -> "Circle":
        return Circle((transform @ np.hstack([self.center, [0]]))[:2], self.radius)
