"""Zirconium based standard materials"""

from isotopes import Cr, Fe, Ni, O, Sn, Zr

from coremaker.materials.mixture import Mixture
from coremaker.materials.util import room_temperature

# Taken from PNNL Compendium Rev 2
zircalloy_2 = Mixture(
    {O: 2.96e-4, Cr: 7.6e-5, Fe: 7.1e-5, Ni: 3.4e-5, Zr: 0.042541, Sn: 4.65e-4}, temperature=room_temperature
)
zircalloy_4 = Mixture({O: 2.95e-4, Cr: 7.6e-5, Fe: 1.41e-4, Zr: 0.04252, Sn: 4.64e-4}, temperature=room_temperature)
