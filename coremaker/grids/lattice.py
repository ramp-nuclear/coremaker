from typing import Any, Callable, Type, TypeVar

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from ramp_core.serializable import Serializable, deserialize_default

from coremaker.geometries import Box, FiniteCylinder, HexPrism, Rectangle
from coremaker.materials.mixture import Mixture as ConcMixture
from coremaker.protocols.grid import Lattice
from coremaker.protocols.mixture import Mixture
from coremaker.surfaces.util import comma_format
from coremaker.transform import Transform
from coremaker.units import cm


class CartesianLattice(Lattice):
    """A cartesian shaped lattice.

    For more explanations on each method, see :class:`Lattice`

    """

    ser_identifier = "CartLattice"

    def __init__(self, center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 dimensions: tuple[cm, cm],
                 height: cm | None,
                 mixture: Mixture,
                 ):
        self.transform = Transform(center)
        self._shape = np.array(shape, dtype=int)
        self.dimensions = np.array(dimensions, dtype=float)
        self.height = height
        self.mixture = mixture

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {"center": self.origin.tolist(),
                                     "shape": self._shape.tolist(),
                                     "dimensions": self.dimensions.tolist(),
                                     "height": self.height,
                                     "mixture": self.mixture.serialize()
                                     }

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        mixture = deserialize_default(d["mixture"], supported=supported, default=ConcMixture)
        return cls(center=tuple(d["center"]),
                   shape=tuple(d["shape"]),
                   dimensions=tuple(d["dimensions"]),
                   height=d["height"],
                   mixture=mixture
                   )

    def __eq__(self, other: "CartesianLattice") -> bool:
        if not isinstance(other, CartesianLattice):
            return NotImplemented
        return (self.transform == other.transform
                and np.all(self._shape == other._shape)
                and np.allclose(self.dimensions, other.dimensions)
                and self.mixture == other.mixture)

    def __hash__(self) -> int:
        return hash((self.transform, tuple(self._shape), tuple(self.dimensions)))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape.item(0), self._shape.item(1)

    shape.__doc__ = Lattice.shape.__doc__

    @property
    def origin(self) -> np.ndarray:
        return self.transform.translation.flatten()

    origin.__doc__ = Lattice.origin.__doc__

    def center(self, index: tuple[int, int]) -> tuple[cm, cm, cm]:
        r"""Get the center of the site at the index relative to the center of the lattice.

        To get the location at index (i, j) you could first go to the center of (0, 0),
        and move from there.
        It is clear that, since `i` is the y index and `j` is the x index:

        .. math:: \vec{v}_{i, j} = \vec{v}_{0,0} + (j \cdot d_x, i \cdot d_y)

        If the shape in (n, m) it can be observed that:

        .. math:: \vec{v}_{0,0} = - (\frac{m \cdot d_x}{2}, \frac{n \cdot d_y}{2})
                  + \frac{1}{2}(d_x, d_y)

        therefore:

        .. math:: \vec{v}_{i, j} &= - (\frac{m \cdot d_x}{2}, \frac{n \cdot d_y}{2})
                  + \frac{1}{2}(d_x, d_y) + (j \cdot d_x, i \cdot d_y) \\
                  &=  \left( \frac{2*j+1-m}{2}d_x, \frac{2*i+1-n}{2}d_y \right)

        Parameters
        ----------
        index: Sequence of ints. Must be of length 3.

        """
        shift = np.zeros(3)
        shift[:-1] = (2 * np.array(index)[::-1] + 1 - self._shape)
        dim3d = np.hstack((self.dimensions, [0.]))
        # Safe because we know shift, dim3d are of length 3
        return tuple((shift * dim3d / 2))  # type: ignore

    @property
    def geometry(self) -> Box | Rectangle:
        dimensions_2d: tuple[float, float] = self.dimensions * self._shape
        if self.height:
            dimensions: tuple[float, float, float] = (*dimensions_2d, self.height)  # type: ignore
            return Box((0.,) * 3, dimensions)
        return Rectangle((0.,) * 2, dimensions_2d)

    geometry.__doc__ = Lattice.geometry.__doc__

    @property
    def inner_geometry(self) -> Box | Rectangle:
        if self.height:
            dimensions_3d: tuple[float, float, float] = (*self.dimensions, self.height)  # type: ignore
            return Box((0., 0., 0.), dimensions_3d)
        return Rectangle((0., 0.), tuple(self.dimensions))

    inner_geometry.__doc__ = Lattice.inner_geometry.__doc__

    def __repr__(self) -> str:
        return f'CartesianLattice<Center: {comma_format(self.origin)}, ' \
               f'Shape: {tuple(self._shape)}, ' \
               f'Dimensions: {comma_format(self.dimensions)}, ' \
               f'Mixture: {self.mixture}>'


def default_hexagonal_index_position(index: tuple[int, int]) -> tuple[float, float]:
    r"""Returns the default position from the center of the lattice in the cartesian x,y coordinates
    normelised by the pitch.
    To get the location at index (i, j) you could first go to the center of (0, 0),
    and move from there.
    Parameters
    ----------
    index: Sequence of ints. Must be of length 2.

    """
    return tuple((index[1] * np.sqrt(3) / 2, index[1] / 2 + index[0]))


class HexagonalLattice(Lattice):
    """A hexagonal shaped lattice.

    For more explanations on each method, see :class:`Lattice`

    Parameters:
    -------------
    center:
        Coordinates of the center of the lattice
    shape: tuple[int, int]
        Maximal number of sites on each index.
        The shape represent the number of sites in the x and y axes.
        The default shape: should be odd numbers and shape[1] >= shape[0] because in the default shaping a tip of the
        hexagon is in the y direction, and it isn't possible mathematically to have shape[1] < shape[0]
    pitch: cm
        Pitch between parallel hexagonal walls, in cm.
    height: cm
            Length along the prism line, in cm.
    outer_radius: cm
             The outer radius in cm of the bounding cylinder of ther hexagonal lattice.
    index_position:
        Function that receives an index and returns the x,y position of the center of the site

    Examples
    --------
    5x3 hexagonal lattice:
      o   o   o   o
    o   o   o   o   o
      o   o   o   o

    >>> from coremaker.materials.water import make_light_water
    >>> hexlat = HexagonalLattice((0, 0, 0), (5, 3), 1, 1, 10, make_light_water(20))
    >>> [[f"{v:.3g}" for v in hexlat.center(index)[:2]] for index in [(-2, 1), (-1, 1), (0, 1), (1, 1)]]
    [['-1.5', '0.866'], ['-0.5', '0.866'], ['0.5', '0.866'], ['1.5', '0.866']]
    >>> [[f"{v:.3g}" for v in hexlat.center(index)[:2]] for index in [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]]
    [['-2', '0'], ['-1', '0'], ['0', '0'], ['1', '0'], ['2', '0']]
    >>> [[f"{v:.3g}" for v in hexlat.center(index)[:2]] for index in [(-1, -1), (0, -1), (1, -1), (2, -1)]]
    [['-1.5', '-0.866'], ['-0.5', '-0.866'], ['0.5', '-0.866'], ['1.5', '-0.866']]

    """

    ser_identifier = "HexLattice"

    def __init__(self,
                 center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 pitch: cm,
                 height: cm,
                 outer_radius: cm,
                 mixture: Mixture,
                 index_position: Callable[[tuple[int, int]], tuple[float, float]] = default_hexagonal_index_position):
        self.transform = Transform(center)
        self._shape = np.array(shape)
        self.pitch = pitch
        self.height = height
        self.mixture = mixture
        self.outer_radius = outer_radius
        self.index_position = index_position

    def serialize(self) -> tuple[str, dict[str, Any]]:
        if self.index_position != default_hexagonal_index_position:
            raise TypeError("We currently do not support serialization for non-default index_position")
        return self.ser_identifier, {"center": self.origin.tolist(),
                                     "shape": [int(v) for v in self._shape.tolist()],
                                     "pitch": self.pitch,
                                     "height": self.height,
                                     "outer_radius": self.outer_radius,
                                     "mixture": self.mixture.serialize(),
                                     }

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        mixture = deserialize_default(d["mixture"], supported=supported, default=ConcMixture)
        return cls(center=tuple(d["center"]),
                   shape=tuple(d["shape"]),
                   pitch=d["pitch"],
                   height=d["height"],
                   outer_radius=d["outer_radius"],
                   mixture=mixture,
                   )

    def __eq__(self, other: "HexagonalLattice") -> bool:
        if not isinstance(other, HexagonalLattice):
            return NotImplemented
        return (self.transform == other.transform
                and np.all(self._shape == other._shape)
                and np.isclose(self.pitch, other.pitch)
                and np.isclose(self.outer_radius, other.outer_radius)
                and self.mixture == other.mixture)

    def __hash__(self) -> int:
        return hash((self.transform, tuple(self._shape), self.pitch, self.outer_radius, self.height))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape.item(0), self._shape.item(1)

    shape.__doc__ = Lattice.shape.__doc__

    @property
    def origin(self) -> np.ndarray:
        return self.transform.translation.flatten()

    origin.__doc__ = Lattice.origin.__doc__

    def center(self, index: tuple[int, int]) -> tuple[cm, cm, cm]:
        r"""Get the center of the site at the index relative to the center of the lattice.

        To get the location at index (i, j) you could first go to the center of (0, 0),
        and move from there.

        The default center coordinates system is of 2 vectors with 60 degrees angle between them/
        The i axis is on the x-axis and the j axis is 60 degrees to the north

        Parameters
        ----------
        index: Sequence of ints. Must be of length 3.

        Examples
        --------
        >>> from coremaker.materials.water import make_light_water
        >>> hex_lattice = HexagonalLattice((0., 0., 0.), (7, 9), 1, 1, 10, make_light_water(20.))
        >>> hex_lattice.center((0, 0))
        (0.0, 0.0, 0.0)

        """

        dim3d = (self.pitch, self.pitch, 0.)
        rule = (index[1] / 2 + index[0], index[1] * np.sqrt(3) / 2, 0.)
        return tuple(float(r * d) for r, d in zip(rule, dim3d))  # type: ignore  # Safe because we made them of length 3

    @property
    def geometry(self) -> FiniteCylinder:
        return FiniteCylinder((0., 0., 0.), self.outer_radius, self.height, (0, 0, 1))

    geometry.__doc__ = Lattice.geometry.__doc__

    @property
    def inner_geometry(self) -> HexPrism:
        return HexPrism((0., 0., 0.), self.pitch, self.height)

    inner_geometry.__doc__ = Lattice.inner_geometry.__doc__

    def __repr__(self) -> str:
        return f'HexagonalLattice<Center: {comma_format(self.origin)}, ' \
               f'Shape: {tuple(self._shape)}, ' \
               f'Pitch: {self.pitch}, ' \
               f'Mixture: {self.mixture}>'
