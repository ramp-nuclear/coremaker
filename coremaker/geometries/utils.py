from functools import reduce

import numpy as np

from coremaker.geometries.box import Box
from coremaker.geometries.holed import ConcreteHoledGeometry
from coremaker.geometries.infinite import _InfiniteGeometry
from coremaker.geometries.union import ConcreteUnionGeometry
from coremaker.protocols.geometry import Geometry


def _bounding_box_union(bbox1: Box, bbox2: Box):
    x0, x1 = bbox1.center - bbox1.dimensions / 2, bbox2.center + bbox2.dimensions / 2
    x0, x1 = np.minimum(x0, x1), np.maximum(x0, x1)
    return Box(0.5 * (x0 + x1), x1 - x0)


def calculate_bounding_box(geometry: Geometry):
    try:
        # Safe because we catch the exception
        return geometry.bounding_box()  # type: ignore
    except AttributeError:
        pass
    match geometry:
        case ConcreteUnionGeometry():
            geometry: ConcreteUnionGeometry
            return reduce(_bounding_box_union, (calculate_bounding_box(g) for g in geometry.geometries))
        case ConcreteHoledGeometry():
            geometry: ConcreteHoledGeometry
            return calculate_bounding_box(geometry.inclusive)
        case _InfiniteGeometry():
            raise ValueError("infinite geometry have no bounding box")
        case _:
            raise TypeError("unknown geometry type")
