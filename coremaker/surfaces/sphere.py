"""A spherical surface.

"""
import math

import numpy as np
from scipy.linalg import norm as norm2

from coremaker.surfaces.util import DECIMAL_PRECISION, comma_format
from coremaker.transform import Transform
from coremaker.units import cm


class Sphere:
    """A spherical surface.

    To understand most methods, see the :class:`~coremaker.protocols.surface.Surface` protocol.
    """

    __slots__ = ['center', 'radius', 'inside']

    def __init__(self,
                 center: tuple[cm, cm, cm],
                 radius: cm,
                 *,
                 inside: bool):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The center of the sphere compared to the origin, in cm.
        radius: cm
            The radius of the sphere, in cm.
        inside: bool
            A flag to know whether this is the internal or external side of the sphere
        """
        self.center = np.round(center, decimals=DECIMAL_PRECISION)
        self.radius = float(np.round(radius, decimals=DECIMAL_PRECISION))
        self.inside = inside

    def __hash__(self):
        return hash((tuple(self.center), self.radius, self.inside))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sphere):
            return NotImplemented
        return (np.all(self.center == other.center) and
                (self.radius == other.radius) and
                self.inside == other.inside)

    def __neg__(self) -> "Sphere":
        return Sphere(self.center, self.radius, inside=not self.inside)

    def transform(self, transform: Transform) -> "Sphere":
        return Sphere(transform @ self.center,
                      self.radius, inside=self.inside)

    def normal(self, point: tuple[cm, cm, cm]) -> tuple[cm, cm, cm]:
        result = np.array(point) - np.array(self.center)
        result /= norm2(result, ord=2)
        if not self.inside:
            result = -result
        return tuple(result)

    def isclose(self, other: "Sphere"):
        if not isinstance(other, Sphere):
            return False
        return ((self.inside == other.inside)
                and (math.isclose(self.radius, other.radius, rel_tol=0, abs_tol=1e-5))
                and all(math.isclose(x, y, rel_tol=0, abs_tol=1e-5)
                        for x, y in zip(self.center, other.center)))

    def __repr__(self) -> str:
        return f"Sphere<Center: {comma_format(self.center)}, Radius: {self.radius:.3e}, " \
               f"{'Internal' if self.inside else 'External'}>"
