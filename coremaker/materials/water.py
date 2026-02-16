"""Tools to generate water mixtures.

Currently, the water density is taken from the WIMS data written in Polytechnique's
DRAGON code. Any preferred way can replace this in the future, but this is
kept for now as backwards compatability with calculations we used to do with
the DRAGON code.

"""
# noinspection NonAsciiCharacters
from itertools import chain
from typing import Callable, TypeVar, Union, Dict, Optional, Iterable

import numpy as np
from cytoolz import unique
from isotopes import Isotope, H, H1, H2 as D, avogadro, O
from scipy.interpolate import interp1d

from coremaker.materials.mixture import Mixture
from coremaker.protocols.mixture import Chemical

SabTables = Optional[Iterable[Chemical]]

__all__ = ['make_water', 'make_light_water', 'make_heavy_water']


# noinspection NonAsciiCharacters
def make_water(*, temp: float, impurities: Optional[Dict[Isotope, float]],
               expand_naturals: bool, main_iso: Isotope, purity: float,
               density_strategy: Callable[[float], float],
               sab_tables: SabTables = None) -> Mixture:
    """Backend tool for creating water mixtures.

    Parameters
    ----------
    temp: deg C
        Water temperature in Celsius.
    impurities: Optional[Dict[Isotope, float]]
        Dictionary of impurities in the water.
    expand_naturals: bool
        Flag for whether to expand natural elements into isotopes
    main_iso : Isotope
        Main isotope of Hydrogen in this water. Used so we know if to use the
        H2O or D2O density functions.
    purity: float
        Fractional purity of H2O in D2O or vice versa.
    density_strategy: func: float->float
        Function used to calculate the water density given temperature.
    sab_tables: Optional[Iterable[Chemical]]
        S_{αβ} tables to associate with the mixture. Defaults to no S_{αβ} treatment.

    See Also
    --------
    :func:`make_light_water`, :func:`make_heavy_water`

    """
    if main_iso not in {H1, D}:
        raise ValueError("Main isotope must be either Hydrogen or Deuterium")
    other_iso = D if main_iso == H1 else H1
    impurities = impurities or {}
    ρ_w0 = density_strategy(temp)
    ρ_w, ρ_w_other = [ρ_w0 * (1. - sum(impurities.values())) * frac
                      for frac in (purity, 1. - purity)]
    mass_per_molecule = (2 * main_iso.mass + O.mass) / avogadro  # In 1e-24 gr
    mass_per_molecule_other = (2 * other_iso.mass + O.mass) / avogadro
    nd_w, nd_w_other = [ρ / mpm for ρ, mpm in zip((ρ_w, ρ_w_other),
                                                  (mass_per_molecule,
                                                   mass_per_molecule_other))]
    wiso = {main_iso: 2 * nd_w, other_iso: 2 * nd_w_other, O: nd_w + nd_w_other}
    aiso = {iso: ρ_w0 * ppm * avogadro / iso.mass
            for iso, ppm in impurities.items() if ppm > 0.}
    isos = {iso: wiso.get(iso, 0.) + aiso.get(iso, 0.)
            for iso in unique(chain(wiso, aiso))}
    mix = Mixture(isos, temp, sab_tables=sab_tables)
    return Mixture.expand(mix) if expand_naturals else mix


def make_light_water(temp: float,
                     impurities: Optional[Dict[Isotope, float]] = None,
                     expand_naturals: bool = False,
                     sab_tables: SabTables = (Chemical.LightWater,
                                              Chemical.HeavyWater)
                     ) -> Mixture:
    """Create a light water mixture at a given temperature.

    One can put in impurities, so the water will contain other isotopes as well.
    The assumption is that these impurities do not change the density of the
    mixture compared with a pure water mixture.

    Parameters
    ----------
    temp: degrees Celcius
        Light water temperature, in degrees Celsius.
    impurities: Optional[Dict[Isotope, float]]
        Dictionary of things to add in, as weight fractions.
    expand_naturals: bool
        Flag for whether to expand any natural elements into their
        naturally occuring isotopes.
    sab_tables: Optional[Iterable[Chemical]]
        S_{αβ} tables to associate with the mixture. Defaults to both light and
        heavy water tables, since there is some heavy water in natural light water.

    See Also
    --------
    :func:`make_water`

    Returns
    -------
    Mixture
        The Mixture object that makes up light water at this temperature in
        otherwise STD conditions.

    """
    return make_water(temp=temp, impurities=impurities,
                      expand_naturals=expand_naturals,
                      purity=1., main_iso=H1,
                      density_strategy=_H2O,
                      sab_tables=sab_tables)


def make_heavy_water(temp: float, purity: float = 1.,
                     impurities: Optional[Dict[Isotope, float]] = None,
                     expand_naturals: bool = False,
                     sab_tables: SabTables = None) -> Mixture:
    """Create a heavy water mixture at a given temperature

    Parameters
    ----------
    temp: degrees Celsius
        Heavy water temperature, in degrees Celsius.
    purity: float
        The purity of heavy water in the water mixture, as a fraction.
    impurities: Optional[Dict[Isotope, float]]
        Dictionary of things to add in, as weight fractions.
    expand_naturals: bool
        Flag for whether to expand any natural elements into their
        naturally occuring isotopes.
    sab_tables: Optional[Iterable[Chemical]]
        S_{αβ} tables to associate with the mixture. Defaults to what
        is appropriate for heavy water that is potentially impure.

    Returns
    -------
    Mixture
        The Mixture object that makes up heavy water at this temperature in
        otherwise STD conditions.

    """
    if sab_tables is None:
        if 0. < purity < 1.:
            sab_tables = (Chemical.HeavyWater, Chemical.LightWater)
        elif purity == 1:
            sab_tables = (Chemical.HeavyWater,)
    return make_water(temp=temp, purity=purity, impurities=impurities,
                      expand_naturals=expand_naturals, main_iso=D,
                      density_strategy=_D2O,
                      sab_tables=sab_tables)


ScalVec = TypeVar('ScalVec', bound=Union[float, np.ndarray])
Interp_Func = Callable[[ScalVec], ScalVec]

# Data taken from the open source DRAGON project, which it took from its original sources:
# AECL 7531, Journal of phys. chem. ref. data, vol 11, 1982
_Dtemp = np.array([3.8, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 49.99, 100.0, 111.02, 150.02, 200.0, 250.0, 275.,
                   300., 325., 350.127, 360.057])
_Ddens_inv = np.array([0.90464, 0.90439, 0.90419, 0.90428, 0.90472, 0.90545, 0.90645, 0.90771, 0.90918, 0.91274,
                       0.94057, 0.94866, 0.98296, 1.04354, 1.13149, 1.19270, 1.2740, 1.3917, 1.6044, 1.7709])

# CRNL-1533
_Htemp = np.array([3.98, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.6, 120.2, 133.5, 143.6, 151.8, 158.8, 165.0,
                   170.4, 175.4, 179.9, 188.0, 195.0, 201.4, 207.1, 212.4, 217.2, 221.8, 226.0, 230.0, 233.8, 237.4,
                   240.9, 244.2, 247.3, 250.3, 253.2, 256.0, 258.8, 261.4, 263.9, 266.4, 268.8, 271.1, 273.3, 275.6,
                   277.7, 279.8, 281.8, 283.8, 285.8, 287.7, 289.6, 291.4, 293.2, 295.0, 296.7, 298.4, 300.1, 301.7,
                   303.3, 307.2, 311.0, 314.6, 318.0, 321.4, 324.6, 327.8, 330.8, 333.8, 336.6, 339.4, 342.1, 344.8,
                   347.3, 349.8, 352.3, 354.6, 357.0, 359.2, 361.4, 363.6, 365.7, 367.8, 369.8, 371.8, 373.7])
_Hdens = np.array([0.999973, 0.998418, 0.995848, 0.992385, 0.988164, 0.983274, 0.977785, 0.971753, 0.965218, 0.958479,
                   0.942737, 0.931621, 0.922712, 0.915136, 0.908466, 0.902459, 0.896961, 0.891869, 0.887107, 0.878373,
                   0.870456, 0.863165, 0.856371, 0.849982, 0.843931, 0.838165, 0.832644, 0.827336, 0.822215, 0.817259,
                   0.812449, 0.807771, 0.803211, 0.798756, 0.794401, 0.790133, 0.785946, 0.781833, 0.777787, 0.773804,
                   0.769879, 0.766007, 0.762183, 0.758405, 0.754669, 0.750972, 0.747310, 0.743682, 0.740085, 0.736516,
                   0.732973, 0.729455, 0.725958, 0.722482, 0.719025, 0.715585, 0.712160, 0.708750, 0.705352, 0.696902,
                   0.688503, 0.680135, 0.671780, 0.663420, 0.655039, 0.646619, 0.638141, 0.629586, 0.620932, 0.612153,
                   0.603220, 0.594098, 0.584739, 0.575085, 0.565162, 0.554167, 0.543564, 0.531796, 0.519245, 0.505736,
                   0.490945, 0.474241, 0.454256, 0.427351, 0.374304])

_H2O = interp1d(_Htemp, _Hdens)
_D2O_inv = interp1d(_Dtemp, _Ddens_inv)

T = TypeVar("T", bound=Union[np.ndarray, float])


def _D2O(temp: T) -> T:
    """D2O density by temperature function

    For some reason, the data is inverted in the source (specific volume is given rather than density).
    For backward compatability reasons, we linearly interp the specific volume and invert rather than interp the
    densities at the given temperatures.

    Parameters
    ----------
    temp: Celsius
        Water temperature.

    Returns
    -------
    Water density, in g/cm^3

    """
    return 1 / _D2O_inv(temp)
