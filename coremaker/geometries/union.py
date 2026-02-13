"""Concrete implementation of the UnionGeometry Protocol.

"""
from typing import Iterable, Sequence

import numpy as np

from coremaker.geometries import Box
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.surface import Surface
from coremaker.transform import Transform


def union_bounding_box(geometries: Sequence[Geometry]) -> Box:
    lower_left = np.min(np.vstack(list(map(lambda x: x.bounding_box().center -
                                                     x.bounding_box().dimensions / 2,
                                           geometries))), axis=0)
    upper_right = np.max(np.vstack(list(map(lambda x: x.bounding_box().center +
                                                      x.bounding_box().dimensions / 2,
                                            geometries))), axis=0)
    center = tuple((lower_left + upper_right) / 2)
    dimensions = tuple(upper_right - lower_left)
    return Box(center, dimensions)


class ConcreteUnionGeometry:
    """A concrete implementation of the UnionGeometry Protocol.

    """

    def __init__(self, geometries: Iterable[Geometry], volume=None):
        self.geometries = tuple(geometries)
        if volume and len(self.geometries) == 1 and not np.isclose(self.geometries[0].volume, volume):
            raise ValueError(f"The set volume of union geometry {volume} is not equal to the volume of the"
                             f"unique geometry inside which is {self.geometries[0].volume}")
        if len(self.geometries) == 1:
            self._volume = self.geometries[0].volume
        else:
            self._volume = volume

    @property
    def volume(self) -> float | None:
        """Get the volume of the union, if known.

        The volume of a union geometry is hard to know, because we don't
        know if the geometries overlap. Thus, the default implementation
        gives None for sequences longer than 1.

        """
        return self._volume

    @volume.setter
    def volume(self, value) -> None:
        """
        The volume of a union geometry is hard to know, because we don't
        know if the geometries overlap. The user can set the volume
        to be a known number.

        """
        self._volume = value

    @property
    def surfaces(self) -> list[Surface]:
        """Return a list of all surfaces, regardless of whom they belong to.

        """
        return list(np.concatenate([list(g.surfaces) for g in self.geometries]))

    def transform(self, transform: Transform) -> "ConcreteUnionGeometry":
        """Apply the transformation to the union.

        Parameters
        ----------
        transform: Transform
            The transformation to apply.

        """
        return ConcreteUnionGeometry(tuple(g.transform(transform) for g in
                                           self.geometries), volume=self.volume)

    def bounding_box(self):
        return union_bounding_box(self.geometries)

    def __eq__(self, other: Geometry) -> bool:
        if isinstance(other, type(self)):
            return (self._volume == other._volume) \
                & (self.geometries == other.geometries)
        return NotImplemented
