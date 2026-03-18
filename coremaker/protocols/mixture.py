"""Protocol for what material mixtures of isotopes/zaids look like.

If possible, other packages should only rely on this protocol. That would allow
for multiple implementations of mixtures down the line, though for now we think
that the concrete Mixture class in :class:`coremaker.protocols.mixture.Mixture`
is general enough for all known use cases.

"""

from enum import Enum, auto
from typing import Hashable, Protocol, Sequence

import numpy as np
from isotopes import ZAID, Isotope
from ramp_core.serializable import Serializable


class Chemical(Enum):
    """Known chemicals that can affect isotopic cross-sections."""

    Be = auto()
    Be_in_BeO = auto()
    O_in_BeO = auto()
    Benzene = auto()
    Graphite = auto()
    LightWater = auto()
    HeavyWater = auto()
    Polyethylene = auto()

    @property
    def isotopes(self):
        """which isotopes this chemical affect"""
        return tuple(
            Isotope.from_name(x)
            for x in {
                "Be": ("Be9", "Be"),
                "Be_in_BeO": ("Be9", "Be"),
                "O_in_BeO": ("O16", "O"),
                "Benzene": ("C", "C12"),
                "Graphite": ("C", "C12"),
                "LightWater": ("H1", "H"),
                "HeavyWater": ("H2", "H"),
                "Polyethylene": ("H1", "H"),
            }[self.name]
        )


class Mixture(Serializable, Hashable, Protocol):
    """An isotopic mixture representation protocol.

    We treat a mixture as an immutable, hashable object. Once any piece of the
    program hands a mixture to another, they should always treat them as immutable,
    even if it is possible to mutate.

    """

    sab: Sequence[Chemical]
    temperature: float
    isotopes: dict[ZAID, float]

    def __bool__(self) -> bool:
        return bool(self.isotopes)

    def __eq__(self, other: "Mixture") -> bool: ...

    def get(self, k: ZAID, default=0.0, /) -> float:
        """Get the density of a specific isotope in the mixture.

        Parameters
        ----------
        k: ZAID
            The isotope to get.
        default: float
            Default density if the isotope isn't there

        """
        return self.isotopes.get(k, default)


def are_close(mix1: Mixture, mix2: Mixture) -> bool:
    """Tests whether two mixtures are close.

    Closeness of mixtures is useful when we want to save repetition due to
    differences that don't actually matter in practice. It differs from equality
    because we want equality only if the hash is equivalent, and closeness does
    not guarantee that equal mixtures have the same hash.

    Parameters
    ----------
    mix1: Mixture
    mix2: Mixture

    """
    return (
        frozenset(mix1.sab) == frozenset(mix2.sab)
        and np.isclose(mix1.temperature, mix2.temperature, rtol=1e-2, atol=1e-2)
        and round_densities(mix1.isotopes) == round_densities(mix2.isotopes)
    )


def round_densities(isotopes: dict[ZAID, float], decimals: int = 12) -> dict[ZAID, float]:
    """Round up the densities in a dictionary, so comparison can be done on the
    values up to the given wanted precision.

    Parameters
    ----------
    isotopes: dict[ZAID, float]
        The dictionary of isotopes whose densities to round.
    decimals: int
        The number of decimal points to use in the rounding.


    Returns
    -------
    dict[ZAID, float]
        A new dictionary with the rounded number densities.

    Examples
    --------
    >>> round_densities({ZAID(92, 235, 0): 1.29999, ZAID(6, 14, 0): 0.0001}, 3)
    {922350: 1.3, 60140: 0.0}
    """
    return {iso: np.round(nd, decimals).item() for iso, nd in isotopes.items()}
