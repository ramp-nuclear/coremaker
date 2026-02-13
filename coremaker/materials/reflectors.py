"""Reflector materials

"""
from isotopes import Be, C, B

from coremaker.materials.mixture import Mixture
from coremaker.materials.util import room_temperature
from coremaker.protocols.mixture import Chemical


# Taken from PNNL Compendium Rev. 2
beryllium = Mixture({Be: 1.23487e-1}, room_temperature, (Chemical.Be,))
# The source says impurities differ and users should pick their own values!
# Use at your own risk!
graphite = Mixture.alloy(C, {B: 1e-6}, 1.7,
                         room_temperature, (Chemical.Graphite,)
                         )
