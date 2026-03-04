"""Finite tube (annulus) geometries.

"""
from typing import Literal

import numpy as np
from ramp_core.serializable import Serializable
from scipy.linalg import norm as norm2

from coremaker.geometries.cylinder import cylinder_bounding_box
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.util import DECIMAL_PRECISION, allclose, comma_format, isclose
from coremaker.transform import Transform
from coremaker.units import cm


class Annulus(Serializable):
    """A finite annulus geometry.

    """
    
    __slots__ = ("center", "inner_radius", "outer_radius", "length", "axis")
    
    ser_identifier = "Annulus"

    def __init__(self,
                 center: tuple[cm, cm, cm],
                 inner_radius: cm,
                 outer_radius: cm,
                 length: cm,
                 axis: tuple[float, float, float]):
        """

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The location of the center compared to the origin in cm.
        inner_radius: cm
            The radius of the inner tube, in cm.
        outer_radius: cm
            The radius of the outer tube, in cm. Must be larger than inner_radius
        length: cm
            The axial length of the tube, in cm.
        axis: tuple[float, float, float]
            The symmetry axis of the annulus.
        Raises
        ------
        ValueError if outer_radius <= inner_radius

        """
        if not outer_radius > inner_radius:
            raise ValueError(f'The outer radius of the tube {outer_radius:.5e} '
                             f'must be larger than the inner radius {inner_radius:.5e}'
                             f'of the tube.')
        self.center = np.round(center, decimals=DECIMAL_PRECISION)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.length = length
        if axis == (0., 0., 0.):
            raise ValueError("Annulus axis is a direction and thus cannot be 0")
        self.axis = axis

    @classmethod
    def cardinal(cls, center: tuple[cm, cm, cm],
                 inner_radius: cm,
                 outer_radius: cm,
                 length: cm,
                 axis: Literal['x', 'y', 'z', 0, 1, 2]):
        """
        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The location of the center compared to the origin in cm.
        inner_radius: cm
            The radius of the inner tube, in cm.
        outer_radius: cm
            The radius of the outer tube, in cm. Must be larger than inner_radius
        length: cm
            The axial length of the tube, in cm.
        axis: Literal['x', 'y', 'z', 0, 1, 2]
            The symmetry axis of the annulus.
            'x' and 0 are shortcuts for the (1,0,0) axis.
            'y' and 1 are shortcuts for the (0,1,0) axis.
            'z' and 2 are shortcuts for the (0,0,1) axis.

        Raises
        ------
        ValueError if outer_radius <= inner_radius

        """
        return cls(center, inner_radius, outer_radius, length, {
            0: (1.0, 0.0, 0.0),
            1: (0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
            'x': (1.0, 0.0, 0.0),
            'y': (0.0, 1.0, 0.0),
            'z': (0.0, 0.0, 1.0), }[axis])

    @property
    def volume(self) -> float:
        return self.length * np.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    def _equitable(self):
        axis = np.asarray(self.axis)
        axis = axis / norm2(axis, ord=2)
        return self.center, axis, self.inner_radius, self.outer_radius, self.length

    def __eq__(self, other) -> bool:
        if not isinstance(other, Annulus):
            return NotImplemented
        scenter, saxis, *srest = self._equitable()
        ocenter, oaxis, *orest = other._equitable()
        return (allclose(scenter, ocenter)
                and isclose(abs(saxis @ oaxis), 1)
                and all(isclose(a, b) for a, b in zip(srest, orest))
                )

    def __hash__(self):
        c, *r = self.equitable()
        return hash((tuple(c), *r))

    def __repr__(self) -> str:
        return (f"Annulus<Center: {comma_format(self.center)}, "
                f"Radii: {comma_format((self.inner_radius, self.outer_radius))}, "
                f"Length: {self.length:.3e}, {comma_format(self.axis)} Axis>")

    @property
    def surfaces(self) -> tuple[Plane, Plane, Cylinder, Cylinder]:
        axis = np.array(self.axis)
        axis = axis / norm2(axis, ord=2)
        center = np.array(self.center)
        a1, a2, a3 = axis
        bp = axis @ center + self.length / 2
        bm = axis @ center - self.length / 2
        return (Plane(a1, a2, a3, bm),
                -Plane(a1, a2, a3, bp),
                Cylinder(tuple(self.center), self.inner_radius, self.axis, inside=False),
                Cylinder(tuple(self.center), self.outer_radius, self.axis, inside=True))

    def transform(self, transform: Transform) -> "Annulus":
        return Annulus(tuple(transform @ self.center),
                       self.inner_radius, self.outer_radius,
                       self.length, axis=tuple(transform.rotmat @ self.axis))

    def bounding_box(self):
        return cylinder_bounding_box(self.center,
                                     self.axis,
                                     self.outer_radius,
                                     self.length)


class Ring(Serializable):
    """A 2d ring geometry.

    """

    ser_identifier = "Ring"

    def __init__(self,
                 center: tuple[cm, cm],
                 inner_radius: cm,
                 outer_radius: cm):
        """

        Parameters
        ----------
        center: tuple of length 2, in cm
            Array of length 3, for the location of the center compared to the origin,
            in cm.
        inner_radius: cm
            The radius of the inner tube, in cm.
        outer_radius: cm
            The radius of the outer tube, in cm. Must be larger than inner_radius

        Raises
        ------
        ValueError if outer_radius <= inner_radius

        """
        if not outer_radius > inner_radius:
            raise ValueError(f'The outer radius of the tube {outer_radius:.5e} '
                             f'must be larger than the inner radius {inner_radius:.5e}'
                             f'of the tube.')
        self.center = np.round(center, decimals=DECIMAL_PRECISION)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    @property
    def volume(self) -> float:
        return np.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    def __eq__(self, other) -> bool:
        if isinstance(other, Ring):
            return (allclose(self.center, other.center)
                    and all(isclose(getattr(self, d), getattr(other, d))
                            for d in ('inner_radius', 'outer_radius'))
                    )
        return NotImplemented

    def __hash__(self):
        return hash((tuple(self.center.flatten()), self.inner_radius, self.outer_radius))

    def __repr__(self) -> str:
        return (f"Ring<Center: {comma_format(self.center)}, "
                f"Radii: {comma_format((self.inner_radius, self.outer_radius))},")

    @property
    def surfaces(self) -> tuple[Cylinder, Cylinder]:
        return (
            Cylinder(tuple(np.hstack([self.center, [0]])), self.inner_radius,
                     axis=(0, 0, 1), inside=False),
            Cylinder(tuple(np.hstack([self.center, [0]])), self.outer_radius,
                     axis=(0, 0, 1), inside=True))

    def transform(self, transform: Transform) -> "Ring":
        return Ring((transform @ np.hstack([self.center, [0]]))[:2],
                    self.inner_radius, self.outer_radius)
