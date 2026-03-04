import itertools as it
from functools import partial
from math import ceil, floor
from typing import Callable, Hashable

import numpy as np
from scipy.linalg import norm

from coremaker.protocols.surface import Surface
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.sphere import Sphere

norm2 = partial(norm, ord=2)
Level = int
SurfaceIndex = int


class SurfaceCache:
    """
    Surfaces are caches per geometry level. Levels are sometimes referred to as "Universes" in some sources,
    but the meaning is just that we can embed some well-defined coordinate structure within a larger one.
    This class represents a collection of surfaces with the following properties:
    1. Every surface in the collection appears only once per level.
       (if surfaces on the same level are close only one of them will appear).
    2. If a surface s is in the collection then -s must not be in the collection.
    3. If a surface is not in the collection but encountered, it is added.
    4. Some surfaces are preferable over their matching negative surface.
    """
    COARSE_PRECISION = 3

    def __init__(self, separate_surfaces: Callable[[Level], bool]):
        """

        Parameters
        ----------
        separate_surfaces - a function that decide if surfaces should be unique to some universe
        """
        self.separate_surface_per_level = separate_surfaces
        self.count = 0
        self.heuristics: dict[Hashable, list] = {}
        self.surfaces_lookup: dict[tuple[Surface, Level], tuple[SurfaceIndex, Surface]] = {}
        self.surfaces: list[Surface] = []

    @classmethod
    def interval(cls, x: float) -> tuple[float, ...]:
        """Return the interval that x is in when rounding x up and down according to the coarse precision

        Parameters
        ----------
        x: float
        """
        factor = (10 ** cls.COARSE_PRECISION)
        xm, xp = floor(x * factor) / factor, ceil(x * factor) / factor
        if xm == xp:
            return xm,
        return xm, xp

    @staticmethod
    def is_preferable(s: Surface) -> bool:
        """Whether a surface s is preferable over -s
        """
        match s:
            case Plane():
                return sum(s.a) >= 0
            case Cylinder() | Sphere():
                return not s.inside
            case _:
                raise TypeError(f'unknown surface type {type(s)}')

    def calculate_keys(self, surface: Surface) -> tuple[Hashable, ...]:
        """Calculate heuristics keys that have the following property:
        If s1 and s2 are surfaces that are close (according to the method isclose)
        then s1 and s2 must share at least one key.
        """
        match surface:
            case Plane():
                surface: Plane
                a = np.array(surface.a)
                n = norm2(a)
                a = a / n
                b = surface.b / n

                return tuple(('p', *item)
                             for item in it.product(self.interval(a[0]),
                                                    self.interval(a[1]),
                                                    self.interval(a[2]),
                                                    self.interval(b)))
            case Cylinder():
                surface: Cylinder
                center = np.array(surface.center)
                axis = np.array(surface.axis)
                axis = axis / norm2(axis)
                center = center - (axis @ center) * axis
                return tuple(('c',
                              surface.inside,
                              *item,
                              frozenset([(x0, x1, x2), (-x0, -x1, -x2)]),
                              )
                             for (*item, x0, x1, x2) in it.product(self.interval(surface.radius),
                                                                   self.interval(center[0]),
                                                                   self.interval(center[1]),
                                                                   self.interval(center[2]),
                                                                   self.interval(axis[0]),
                                                                   self.interval(axis[1]),
                                                                   self.interval(axis[2]),
                                                                   )
                             )
            case Sphere():
                surface: Sphere
                return tuple(('s', surface.inside, *item)
                             for item in it.product(self.interval(surface.radius),
                                                    self.interval(surface.center[0]),
                                                    self.interval(surface.center[1]),
                                                    self.interval(surface.center[2]),
                                                    ))
            case _:
                raise TypeError(f'unknown surface type {type(surface)}')

    def add_surface(self, surface: Surface, level: Level) -> SurfaceIndex:
        """Add a surface to the collection and return its index.
        Negative index means that the negative of the surface is in the collection.

        Parameters
        ----------
        surface - surface to add
        level - level that it's in

        Returns
        -------
        index of surface in the collection

        """
        if self.is_preferable(m_surface := -surface):
            surface = m_surface
            sign = -1
        else:
            sign = 1
        for key in self.calculate_keys(surface):
            self.heuristics.setdefault(key, []).append(surface)
        self.count += 1
        self.surfaces_lookup[(surface, level)] = self.count, surface
        self.surfaces.append(surface)
        return sign * self.count

    def find_surface_by_key(self, surface: Surface, level: Level) -> tuple[SurfaceIndex, Surface] | None:
        """Find if a surface is in the cache by comparing against surfaces with matching keys.
        Returns the index and surface, if possible, or None if not found.
        Parameters
        ----------
        surface - surface to search for
        level - level to search for surface

        Returns
        -------
        The index of the surface and the surface. None if not found.
        """
        for key in self.calculate_keys(surface):
            for s in self.heuristics.get(key, tuple()):
                if surface.isclose(s):
                    try:
                        ind, indexed_surface = self.surfaces_lookup[(s, level)]
                    except KeyError:
                        pass
                    else:
                        self.surfaces_lookup[(surface, level)] = ind, s
                        return ind, s
        return None

    def find_surface(self, surface: Surface, level: Level, error: bool = False) -> tuple[int, Surface]:
        """Return the index of the surface in the collection.
        If it is not there, add it and then return the index.
        """
        lev = level if self.separate_surface_per_level(level) else 0
        if x := self.surfaces_lookup.get((surface, lev)):
            return x
        m_surface = -surface
        if x := self.surfaces_lookup.get((m_surface, lev)):
            ind, s = x
            return -ind, s
        if x := self.find_surface_by_key(surface, lev):
            return x
        if x := self.find_surface_by_key(m_surface, lev):
            ind, s = x
            return -ind, s
        if error:
            raise LookupError('Surface not found.')
        index = self.add_surface(surface, lev)
        return index, -surface if index < 0 else surface
