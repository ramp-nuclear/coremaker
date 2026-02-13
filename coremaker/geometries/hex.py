"""Hexagonal geometries

"""
from math import sqrt

import numpy as np

from coremaker.geometries.box import Box
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.util import DECIMAL_PRECISION, allclose, isclose, comma_format
from coremaker.transform import Transform
from coremaker.units import cm


class HexPrism:
    """Represent a hexagonal prism
    """

    def __init__(self, center: tuple[cm, cm, cm], pitch: float, height: float):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The center relative to the origin, in cm.
        pitch: cm
            Pitch between parallel hexagonal walls, in cm.
        height: cm
            Length along the prism line, in cm.
        """
        self.center = center
        self.pitch = pitch
        self.height = height

    @property
    def surfaces(self) -> tuple[Plane, Plane, Plane, Plane,
                                Plane, Plane, Plane, Plane]:
        x0, y0, z0 = self.center
        h = self.height
        a = self.pitch
        return (-Plane(1, 0, 0, x0 + a / 2), Plane(1, 0, 0, x0 - a / 2),
                -Plane(0.5, sqrt(3) / 2, 0, 1/2 * a + 0.5 * x0 + sqrt(3) / 2 * y0),
                Plane(0.5, sqrt(3) / 2, 0, -1/2 * a + 0.5 * x0 + sqrt(3) / 2 * y0),
                Plane(0.5, -sqrt(3) / 2, 0, -1/2 * a + 0.5 * x0 - sqrt(3) / 2 * y0),
                -Plane(0.5, -sqrt(3) / 2, 0, 1/2 * a + 0.5 * x0 - sqrt(3) / 2 * y0),
                -Plane(0, 0, 1, z0 + h / 2),
                Plane(0, 0, 1, z0 - h / 2)
                )

    @property
    def volume(self) -> float:
        return self.pitch ** 2 * sqrt(3) * self.height

    def transform(self, transform: Transform) -> "HexPrism":
        if not transform:
            raise NotImplementedError("Transforming HexPrisms is currently unsupported")
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, HexPrism):
            return (allclose(self.center, other.center) and
                    all(isclose(getattr(self, d), getattr(other, d))
                        for d in ('pitch', 'height'))
                    )
        return NotImplemented

    def __repr__(self) -> str:
        return f"HexPrism<Center: {comma_format(self.center)}, " \
               f"Pitch: {self.pitch:.3e}, Height: {self.height:.3e}>"

    def bounding_box(self):
        dimensions = np.array([self.pitch, self.pitch * sqrt(3) / 2, self.length])
        return Box(tuple(self.center), tuple(dimensions))

class Hexagon:
    """
    represent a 2d hexagon
    """

    def __init__(self, center: np.ndarray, pitch: float):
        """

        Parameters
        ----------
        center: Array of 3 lengths
            The center relative to the origin, in cm.
        pitch: cm
            Pitch between parallel hexagonal walls, in cm.
        """
        self.center = center
        self.pitch = pitch


    @property
    def surfaces(self):
        x0, y0 = self.center
        a = self.pitch
        return (-Plane(1, 0, 0, x0 + a / 2), Plane(1, 0, 0, x0 - a / 2),
                -Plane(0.5, sqrt(3) / 2, 0, 1 / 2 * a + 0.5 * x0 + sqrt(3) / 2 * y0),
                Plane(0.5, sqrt(3) / 2, 0, -1 / 2 * a + 0.5 * x0 + sqrt(3) / 2 * y0),
                Plane(0.5, -sqrt(3) / 2, 0, -1 / 2 * a + 0.5 * x0 - sqrt(3) / 2 * y0),
                -Plane(0.5, -sqrt(3) / 2, 0, 1 / 2 * a + 0.5 * x0 - sqrt(3) / 2 * y0),
                )

    @property
    def volume(self) -> float:
        return self.pitch ** 2 * np.sqrt(3)

    def transform(self, transform: Transform) -> "Hexagon":
        if transform:
            raise NotImplementedError("Transforming Hexagons is currently unsupported")
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, Hexagon):
            return (allclose(self.center, other.center) and
                    isclose(self.pitch,other.pitch))
        return NotImplemented

    def __repr__(self) -> str:
        return f"Hexagon<Center: {comma_format(self.center)}, " \
               f"Pitch: {self.pitch:.3e}>"
