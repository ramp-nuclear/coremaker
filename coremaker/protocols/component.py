from typing import Hashable, Protocol

from coremaker.protocols.geometry import Geometry, HoledGeometry, UnionGeometry
from coremaker.protocols.mixture import Mixture


class Component(Protocol, Hashable):
    """All a physical thing has to do in order to be used in adapters.

    Parameters
    ----------
    mixture: Mixture
        The material this is made out of.
    geometry: Geometry or HoledGeometry or UnionGeometry
        The external surfaces of this component, after holes are taken into account.

    """

    mixture: Mixture
    geometry: HoledGeometry | Geometry | UnionGeometry
