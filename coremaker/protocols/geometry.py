"""General Protocols for geometry objects.

In order for most programs to be able to handle different geometries,
it is best if they can avoid knowing about specific geometries and just use these
protocols.
"""

from typing import Hashable, Protocol, Sequence, runtime_checkable

from ramp_core.serializable import Serializable

from coremaker.protocols.surface import Surface
from coremaker.transform import Transform


@runtime_checkable
class Geometry(Serializable, Hashable, Protocol):
    """A protocol for anything to be considered a geometry for the purposes of defining objects

    """

    @property
    def surfaces(self) -> Sequence[Surface]:
        """A property to get a sequence of all the surfaces that make up the geometry.

        """
        return ()

    @property
    def volume(self) -> float | None:
        """The volume within the geometry.

        """
        return None

    def transform(self, transform: Transform) -> "Geometry":
        """Transform the geometry to get all the surfaces changed.

        Parameters
        ----------
        transform: Transform
            The transform to apply

        """
        ...


@runtime_checkable
class HoledGeometry(Serializable, Protocol):
    """A geometry that is defined with some surfaces excluded.

    Parameters
    ----------
    inclusive: Geometry
        The geometry things lie totally within
    exclusives: Sequence[Geometry]
        The holes in the cheese.

    """

    inclusive: Geometry
    exclusives: Sequence[Geometry]

    @property
    def volume(self) -> float | None:
        """The volume within the geometry.

        """
        raise NotImplementedError("Not implemented at the protocol level.")

    @property
    def surfaces(self) -> Sequence[Surface]:
        """Return a list of all surfaces, exclusive or inclusive.

        """
        raise NotImplementedError("Not implemented at the protocol level")

    def __sub__(self, other: Geometry) -> "HoledGeometry":
        raise NotImplementedError("Not implemented at the protocol level.")

    def transform(self, transform: Transform) -> "HoledGeometry":
        """Transform the geometry and all the holes inside.

        Parameters
        ----------
        transform: Transform
            The transformation to apply.

        """
        raise NotImplementedError("Not implemented at the protocol level.")


@runtime_checkable
class UnionGeometry(Serializable, Protocol):
    """A geometry that is defined as any area within the union of some geometries

    """

    geometries: Sequence[Geometry]

    @property
    def volume(self) -> float | None:
        """Get the volume of the union, if known."""
        raise NotImplementedError("Not implemented at the protocol level.")

    @property
    def surfaces(self) -> list[Surface]:
        """Return a list of all surfaces, regardless of whom they belong to.

        """
        raise NotImplementedError("Not implemented at the protocol level.")

    def transform(self, transform: Transform) -> "UnionGeometry":
        """Apply the transformation to the union.

        Parameters
        ----------
        transform: Transform
            The transformation to apply.

        """
        raise NotImplementedError("Not implemented at the protocol level.")
