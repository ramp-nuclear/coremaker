"""A basic 3D box shape.

"""
import itertools as it
from typing import Any, Type, TypeVar, Union

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from ramp_core.serializable import Serializable

from coremaker.surfaces.plane import Plane
from coremaker.surfaces.util import DECIMAL_PRECISION, allclose, comma_format
from coremaker.transform import Transform, identity
from coremaker.units import cm

T = TypeVar("T", bound=(Union["Box", "Rectangle"]))


def _deser(cls: Type[T], d: dict[str, Any], *_, **__) -> T:
    transform = Transform.deserialize(d["transform"])
    return cls(center=tuple(0. for _ in range(cls._dim)),
               dimensions=d["dimensions"],
               transform=transform)

def _volume(self: T) -> float:
    return np.prod(self.dimensions).item(0)

class Box(Serializable):
    """
    represents an axis-parallel 3d box with center and dimensions.
    """

    ser_identifier = "BoxGeo"
    _dim = 3

    def __init__(self, center: tuple[cm, cm, cm] | np.ndarray,
                 dimensions: tuple[cm, cm, cm] | np.ndarray,
                 transform: Transform = identity,
                 ):
        """A box object.

        This object follows the :class:`~coremaker.protocols.geometry.Geometry`
        protocol.

        Parameters
        ----------
        center: tuple[cm, cm, cm]
            The center of the box compared to the origin, in cm.
        dimensions: tuple[cm, cm, cm]
            The dimensions of the box in the x, y and z axes.
        transform: Transform
            The transformation to apply on a box at that center to get the right box.
        """
        self.transform_ = transform @ Transform(translation=center)
        self.dimensions = np.round(dimensions, DECIMAL_PRECISION)

    @classmethod
    def from_vertices(cls: Type[Self], 
                      lower_left: tuple[cm, cm, cm] | np.ndarray,
                      upper_right: tuple[cm, cm, cm] | np.ndarray,
                      transform: Transform = identity) -> Self:
        """Construct a box based on its minimal and maximal (x, y, z) vertices.

        Parameters
        ----------
        lower_left: tuple[cm, cm, cm] | np.ndarray
            The coordinates of the minimal (x, y, z) vertex
        upper_right: tuple[cm, cm, cm] | np.ndarray
            The coordinates of the maximal (x, y, z) vertex
        transform: Transform
            The transformation to apply on a box at that center to get the right box.

        """
        ll, ur = map(np.asarray, (lower_left, upper_right))
        center = tuple((ll + ur) / 2)
        dimensions = tuple(ur - ll)
        return cls(center, dimensions, transform)

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, dict(dimensions=self.dimensions.tolist(),
                                         transform=self.transform_.serialize()
                                         )

    deserialize = classmethod(_deser)

    @property
    def center(self) -> np.ndarray:
        return self.transform_.translation.flatten()

    @property
    def surfaces(self) -> tuple[Plane, Plane, Plane, Plane, Plane, Plane]:
        x_min, x_max = -self.dimensions / 2, self.dimensions / 2
        planes = (-Plane(1, 0, 0, x_max[0]), Plane(1, 0, 0, x_min[0]),
                  -Plane(0, 1, 0, x_max[1]), Plane(0, 1, 0, x_min[1]),
                  -Plane(0, 0, 1, x_max[2]), Plane(0, 0, 1, x_min[2]))
        return tuple(p.transform(self.transform_) for p in planes)  # type: ignore

    @property
    def volume(self) -> float:
        return _volume(self)

    def transform(self, transform: Transform) -> "Box":
        return Box((0.0, 0.0, 0.0),
                   self.dimensions,
                   transform @ self.transform_)

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return (other.transform_ == self.transform_
                    and allclose(other.dimensions, self.dimensions))
        return NotImplemented

    def __hash__(self):
        return hash((self.transform_, tuple(self.dimensions)))

    def __repr__(self) -> str:
        return (f"Box<Transform: {self.transform_}, "
                f"Dimensions: {comma_format(self.dimensions)}>")

    def bounding_box(self):
        edges = np.array(list(it.product(*zip(-self.dimensions / 2,
                                              self.dimensions / 2))))
        tr_edges = (np.hstack((edges, np.ones((8, 1)))
                              ) @ self.transform_.matrix.T)[:, :-1]
        x0, x1 = np.min(tr_edges, axis=0), np.max(tr_edges, axis=0)
        return Box(tuple(0.5 * (x0 + x1)), tuple(x1 - x0))


class Rectangle(Serializable):
    """A rectangle is a box that someone forgot to make finite in z.

    It's lazy but it works.

    See Also
    --------
    :class:`~coremaker.geometries.box.Box`

    """

    ser_identifier = "Rectangle"
    _dim = 2

    def __init__(self,
                 center: tuple[cm, cm],
                 dimensions: tuple[cm, cm],
                 transform: Transform = identity):
        self.transform_ = transform @ Transform(translation=np.hstack([center, [0]]))
        self.dimensions = np.round(dimensions, DECIMAL_PRECISION)

    serialize = Box.serialize
    deserialize = classmethod(_deser)

    @property
    def surfaces(self) -> tuple[Plane, Plane, Plane, Plane]:
        """Gets the surfaces of the rectangle. The z-axis has no surfaces.

        See Also
        --------
        :method:`Box.surfaces`

        """
        x_min, x_max = -self.dimensions / 2, self.dimensions / 2
        planes = (-Plane(1, 0, 0, x_max[0]), Plane(1, 0, 0, x_min[0]),
                  -Plane(0, 1, 0, x_max[1]), Plane(0, 1, 0, x_min[1]))
        return tuple(p.transform(self.transform_) for p in planes)  # type: ignore

    __eq__ = Box.__eq__
    __hash__ = Box.__hash__

    @property
    def volume(self) -> float: return _volume(self)

    def __repr__(self):
        # noinspection PyTypeChecker
        return Box.__repr__(self).replace('Box', 'Rectangle')

    def transform(self, transform: Transform) -> "Rectangle":
        return Rectangle((0., 0.), self.dimensions, transform @ self.transform_)

    @property
    def center(self) -> np.ndarray:
        return self.transform_.translation.flatten()[:2]

    def bounding_box(self):
        dimensions=np.hstack((self.dimensions, [0]))
        edges = np.array(list(it.product(*zip(-dimensions / 2,
                                              dimensions / 2))))
        tr_edges = (np.hstack((edges, np.ones((8, 1)))
                              ) @ self.transform_.matrix.T)[:, :-1]
        x0, x1 = np.min(tr_edges, axis=0), np.max(tr_edges, axis=0)
        return Rectangle(tuple(0.5 * (x0 + x1))[:2], tuple(x1 - x0)[:2])

