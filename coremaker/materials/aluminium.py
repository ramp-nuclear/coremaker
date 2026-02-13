"""Common nuclear aluminium alloys.

"""

from isotopes import Al, Mg, Si, Ti, Cr, Zn, Cu, Fe, Mn

from coremaker.materials.mixture import Mixture
from coremaker.materials.util import room_temperature


# Taken from PNNL Compendium Rev 2
al1050 = aluminium = Mixture({Al: 0.060238}, room_temperature)
al6061 = Mixture({Al: 0.058575,
                  Mg: 6.69e-4, Si: 3.47e-4, Ti: 3e-5, Cr: 6.1e-5,
                  Mn: 2.6e-5,  Fe: 1.19e-4, Cu: 7e-5, Zn: 3.6e-5},
                 room_temperature)
