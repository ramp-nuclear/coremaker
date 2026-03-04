"""A protocol for general representation of a core creation surface object.

It is best that most code use only this protocol. Some programs, however, need to know specific things about surfaces.
For example, most Monte Carlo adapters need to know the type of the surface to be able to model it in a useful manner.

Thus, the :ref:`surfaces` subpackage of CoreMaker is also considered public API, and not just this protocol.
It is still best to make those specific assumptions as local as possible, so new and exciting surfaces can be used.

"""
from typing import Hashable, Protocol

from ramp_core.serializable import Serializable

from coremaker.transform import Transform
from coremaker.units import cm


class Surface(Serializable, Hashable, Protocol):
    """Representation of a surface

    """

    def __neg__(self) -> "Surface":
        """Returns the inverse side of this surface.

        """
        ...

    def isclose(self, other: "Surface"):
        """Tests whether this surface is close enough to another surface of the same type.

        Surfaces are considered close if their parameters are close enough.
        This semantic is a non-transitivie "equality" operation.
        It checks whether two surfaces are in a small enough neighborhood.

        """

    def transform(self, transform: Transform) -> "Surface":
        """Return a shifted and rotated surface given a transform.

        Parameters
        ----------
        transform: Transform
            Affine transformation to apply

        """
        ...

    def normal(self, point: tuple[cm, cm, cm]) -> tuple[float, float, float]:
        """The normal to a point on the surface.

        For a point on the surface return the unit vector that is normal to
        the surface and points out of the surface from that point.

        Parameters
        ----------
        point: tuple[cm, cm, cm]
            The point on the surface to find the outward normal to.

        Returns
        -------
        tuple[float, float, float]
            A normalized direction vector normal to the point at the surface.

        """
