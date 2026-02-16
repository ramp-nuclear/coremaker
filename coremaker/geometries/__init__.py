"""Subpackage that deals with different geometry types.

"""

from .annulus import Annulus, Ring
from .ball import Ball, Circle
from .bare import BareGeometry
from .box import Box, Rectangle
from .cylinder import FiniteCylinder
from .hex import HexPrism, Hexagon
from .infinite import infiniteGeometry
from .union import ConcreteUnionGeometry

jsonable = [BareGeometry, type(infiniteGeometry), Annulus, Ball, Box, 
            FiniteCylinder, HexPrism, ConcreteUnionGeometry, Ring, Circle,
            Rectangle, Hexagon]
serialization_identifiers = {c.ser_identifier: c for c in jsonable}

