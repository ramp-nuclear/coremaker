"""A concrete implementation of the Component Protocol."""

from dataclasses import dataclass

from coremaker.protocols.geometry import Geometry, HoledGeometry, UnionGeometry
from coremaker.protocols.mixture import Mixture


@dataclass(frozen=True)
class ConcreteComponent:
    """A concrete implementation of the :class:`Component` protocol.

    This is the one used by this package. Other packages can have other
    implementations, so long as they fit the :class:`Component` protocol.

    Parameters
    ----------
    mixture: Mixture
        The material this is made out of.
    geometry: HoledGeometry or Geometry
        The external surfaces of this component, after holes are taken into account.

    """

    mixture: Mixture
    geometry: UnionGeometry | HoledGeometry | Geometry

    def __hash__(self):
        return hash(
            (
                hash(self.mixture),
                tuple(hash(s) for s in self.geometry.surfaces),
            )
        )
