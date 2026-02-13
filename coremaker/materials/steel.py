"""Steel based materials

"""
from isotopes import C, Mn, P, S, Fe, Ni, Cr, Si, Mo

from coremaker.materials.mixture import Mixture
from coremaker.materials.util import room_temperature


# Taken from PNNL Compendium Rev 2
steel_304L = Mixture({C: 3.21e-4, Mn: 1.754e-3, P: 7e-5, S: X.XX-5,
                      Si: 1.715e-3, Cr: 1.7605e-2, Ni: 7.798e-3, Fe: 0.058961},
                     temperature=room_temperature)
steel_316L = Mixture({C: 1.2e-4, Mn: 1.754e-3, P: 7e-5, S: X.XX-5,
                      Si: 1.715e-3, Cr: 1.5751e-2, Ni: 9.85e-3, Mo: 1.255e-3,
                      Fe: 0.056416}, temperature=room_temperature)
