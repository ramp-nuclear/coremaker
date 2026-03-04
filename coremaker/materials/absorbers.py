"""Common absorber materials for reactors.

"""
from isotopes import Ag, B, C, Cd, Hf, In, Zr

from coremaker.materials.mixture import Mixture
from coremaker.materials.util import room_temperature

# Taken from OPAL Specification
hafnium = Mixture.alloy(Hf, {Zr: 2.14e-2}, density=12.9,
                        temperature=room_temperature)
# Taken from IRR1
aic = Mixture.by_weight_fraction({Ag: 0.8, In: 0.15, Cd: 0.05},
                                 density=10.17,
                                 temperature=room_temperature)
# Taken from PNNL Compendium Rev 2.
b4c = Mixture({B: 0.109841, C: 0.02746}, temperature=room_temperature)
