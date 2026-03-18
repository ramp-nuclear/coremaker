"""Gaseous mixtures."""

from typing import Iterable

from isotopes import ZAID
from scipy.constants import Boltzmann as kb
from scipy.constants import atmosphere, zero_Celsius

from coremaker.materials import Mixture
from coremaker.protocols.mixture import Chemical

Atmosphere = float
DegreesCelsius = float

m3_in_cmbarn = 1e30


def ideal_gas_mixture(
    particle: dict[ZAID, float],
    pressure: Atmosphere,
    temperature: DegreesCelsius,
    sab_tables: Iterable[Chemical] = (),
) -> Mixture:
    """Creates a mixture with a gas particle that has the correct number density
    for the given temperature and pressure.

    Parameters
    ----------
    particle: dict[ZAID, float]
        A dictionary of how many ZAIDs are in the particle on average.
    pressure: Atmosphere
        Pressure of the gas in atmospheres.
    temperature: degrees Celsius
        The temperature in Celsius.
    sab_tables: Iterable[Chemical]
        An iterable of Chemicals, whose Sab we should consider in this mixture.

    """
    number_density = pressure * atmosphere / ((m3_in_cmbarn * kb) * (temperature + zero_Celsius))
    return Mixture({iso: val * number_density for iso, val in particle.items()}, temperature, sab_tables=sab_tables)
