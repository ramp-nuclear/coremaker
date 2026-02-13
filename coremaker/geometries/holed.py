"""A concrete implementation of the holed geometry protocol

"""
from typing import Sequence

from coremaker.protocols.geometry import Geometry
from coremaker.protocols.surface import Surface
from coremaker.transform import Transform


class ConcreteHoledGeometry:
    """A geometry that is defined with some surfaces excluded.
    """

    __slots__ = ['inclusive', 'internal_exclusives', 'external_exclusives', 'exclusives']

    def __init__(self, inclusive: Geometry,
                 internal_exclusives: Sequence[Geometry],
                 external_exclusive: Sequence[Geometry] = ()):
        """
        Parameters
        ----------
        inclusive: Geometry
            The geometry things lie totally within
        internal_exclusives: Sequence[Geometry]
            The holes in the cheese.
        external_exclusive: Sequence[Geometry]
            The holes in the cheese that are sticking out of the cheese.
        """
        self.inclusive = inclusive
        self.internal_exclusives = internal_exclusives
        self.external_exclusives = external_exclusive
        self.exclusives = list(self.internal_exclusives) + list(self.external_exclusives)

    @property
    def volume(self) -> float | None:
        """The volume within the geometry.

        """
        if self.external_exclusives:
            return None
        try:
            return self.inclusive.volume - sum(g.volume for g in self.exclusives)
        except TypeError:
            return None

    @property
    def surfaces(self) -> list[Surface]:
        """Return a list of all surfaces, exclusive or inclusive.

        """
        return (list(self.inclusive.surfaces)
                + sum((list(g.surfaces) for g in self.exclusives), []))

    def __repr__(self) -> str:
        return f"HoledGeometry({self.inclusive}, {self.exclusives})"

    def __sub__(self, other: Geometry) -> "ConcreteHoledGeometry":
        return type(self)(self.inclusive, list(self.internal_exclusives) + [other], self.external_exclusives)

    def transform(self, transform: Transform) -> "ConcreteHoledGeometry":
        """Transform the geometry and all the holes inside.

        Parameters
        ----------
        transform: Transform
            The transformation to apply.

        """
        return type(self)(self.inclusive.transform(transform),
                          [g.transform(transform) for g in self.internal_exclusives],
                          [g.transform(transform) for g in self.external_exclusives])
