"""Concrete implementations for the Grid and Lattice protocols.

"""
from itertools import product, pairwise
from string import ascii_uppercase
from typing import Iterable, Sequence, Callable

import numpy as np
from more_itertools import prepend
from numpy import append

from coremaker.geometries.box import Box, Rectangle
from coremaker.geometries.cylinder import FiniteCylinder
from coremaker.geometries.hex import HexPrism
from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid, Site, Lattice
from coremaker.protocols.mixture import Mixture
from coremaker.surfaces.util import comma_format
from coremaker.transform import Transform
from coremaker.units import cm


class CartesianLattice:
    """A cartesian shaped lattice.

    For more explanations on each method, see :class:`Lattice`

    """

    def __init__(self, center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 dimensions: tuple[cm, cm],
                 height: cm,
                 mixture: Mixture,
                 ):
        self.transform = Transform(center)
        self._shape = np.array(shape)
        self.dimensions = np.array(dimensions)
        self.height = height
        self.mixture = mixture

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
        return tuple(self._shape)

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
        return tuple((shift * dim3d / 2))

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


class HexagonalLattice:
    """A hexagonal shaped lattice.

    For more explanations on each method, see :class:`Lattice`

    Parameters:
    -------------
    center: coordinates of the center of the lattice
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
    index_position: a function that receive a index in 2D tuple and return the x,y position site of the center of the
    site
    """

    def __init__(self, center: tuple[cm, cm, cm],
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
        return tuple(self._shape)

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

        """

        dim3d = np.array([self.pitch, self.pitch, 0])
        return tuple((index[1] * np.sqrt(3) / 2, index[1] / 2 + index[0], 0)) * dim3d

    @property
    def geometry(self) -> FiniteCylinder:
        return FiniteCylinder((0.,) * 3, self.outer_radius, self.height, (0, 0, 1))

    geometry.__doc__ = Lattice.geometry.__doc__

    @property
    def inner_geometry(self) -> HexPrism:
        return HexPrism(np.zeros(3, np.float64), self.pitch, self.height)

    inner_geometry.__doc__ = Lattice.inner_geometry.__doc__

    def __repr__(self) -> str:
        return f'HexagonalLattice<Center: {comma_format(self.origin)}, ' \
               f'Shape: {tuple(self._shape)}, ' \
               f'Pitch: {self.pitch}, ' \
               f'Mixture: {self.mixture}>'


alphabet = ascii_uppercase


def _sites(shape: tuple[int, int]) -> Iterable[Site]:
    """
    The shape is given in the format of:
    (sites in x direction, sites in y direction).

    Parameters
    ----------
    shape: tuple[int, int]
        Number of rows and columns

    """
    rows, columns = shape[::-1]
    return (f'{letter}{number}' for letter, number in product(alphabet[:rows], range(1, columns + 1)))


class NullGrid:
    """
    Empty Grid with no sites
    """

    def __init__(self):
        pass

    @property
    def lattices(self) -> Sequence[Lattice]:
        return []

    def sites(self) -> Iterable[Site]:
        return []

    def keys(self) -> Iterable[Site]:
        return []

    def values(self) -> Iterable[Element]:
        return []

    def items(self) -> Iterable[tuple[str, Element]]:
        return []


# noinspection PyMissingDocstring
class CartesianGrid(Grid):
    """A grid that just has one cartesian lattice.

    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    def __init__(self, center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 dimensions: tuple[cm, cm],
                 height: cm,
                 mixture: Mixture,
                 rod_contents: dict[Site, Element] | None = None,
                 ):
        self.contents = rod_contents or {}
        self.lattice = CartesianLattice(center, shape, dimensions, height, mixture)

    @property
    def lattices(self) -> tuple[CartesianLattice]:
        return self.lattice,

    lattices.__doc__ = Grid.lattices.__doc__

    def site_index(self, site: Site) -> tuple[CartesianLattice, tuple[int, int]]:
        return self.lattice, (alphabet.index(site[0]), int(site[1:]) - 1)

    site_index.__doc__ = Grid.site_index.__doc__

    def __repr__(self):
        return f"CartesianGrid<Lattice: {self.lattice}>"

    def __len__(self): return len(self.contents)

    def sites(self) -> Iterable[Site]:
        return _sites(self.lattice.shape)

    def keys(self) -> Iterable[Site]:
        return self.contents.keys()

    keys.__doc__ = dict.keys.__doc__

    def values(self) -> Iterable[Element]:
        return self.contents.values()

    values.__doc__ = dict.values.__doc__

    def items(self) -> Iterable[tuple[Site, Element]]:
        yield from self.contents.items()

    items.__doc__ = dict.items.__doc__

    def __getitem__(self, key: Site) -> Element:
        return self.contents[key]

    def __delitem__(self, key: Site):
        del self.contents[key]

    def __contains__(self, item):
        return item in self.contents

    def __setitem__(self, key: Site, value: Element) -> None:
        self.contents[key] = value

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all((self.lattice == other.lattice,
                        self.contents == other.contents))
        return False


def default_serial_to_lattice_index(site_index: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    site_pos = tuple((site_index[0] - shape[0] // 2, int(site_index[1] - shape[1] // 2)))
    return site_pos


class HexagonalGrid(Grid):
    """A grid that just has one hexagonal lattice.
    Rows are named by uppercase letters and columns by numbers.

    Parameters
    ----------
    center: The center of the grid in a 3D space in cm
    shape: the shape of the hexagon in the default coordinate system see: default_hexagonal_index_position
    pitch: the pitch of each site inside the hexagonal grid in cm
    height: the height of the grid in cm
    outer_radius:  the bounding radius of the grid in cm
    mixture: the default mixture inside each site
    rod_contents: the rods that are loaded inside the grid
    prefix_indexing: this is a sequence that maps site's name prefix to a index from 0 to shape[0] - 1
    suffix_indexing: this is a sequence that maps site's name suffix to a index from 0 to shape[0] - 1
    site_name_partition: a function that split the site name to prefix and suffix, it is here because not always
    the split is trivial. the default that assumed is letter + number as a name
    serial_to_lattice_index: maps between the serial index of the site to the lattice index
    Here is an example of the process that a site goes from Site object to lattice index
    >>> from coremaker.materials.water import make_light_water
    >>> h_grid = HexagonalGrid((1, 1, 1), (3, 3), 1, 1, 1, make_light_water(50))
    >>> h_grid._site_name_partition("A1")
    ('A', '1')
    >>> (h_grid.prefix_indexing.index('A'), h_grid.suffix_indexing.index('1'))
    (0, 0)
    >>> h_grid._serial_to_lattice_index((0, 0),(3, 3))
    (-1, -1)
    >>> h_grid.site_index("A1")[1]
    (-1, -1)
    """

    def __init__(self, center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 pitch: cm,
                 height: cm,
                 outer_radius: cm,
                 mixture: Mixture,
                 rod_contents: dict[Site, Element] | None = None,
                 prefix_indexing: Sequence = alphabet,
                 suffix_indexing: Sequence | None = None,
                 site_name_partition: Callable[[str], tuple[int, int]] = lambda site_name: (site_name[0], site_name[1:]),
                 serial_to_lattice_index: Callable[
                     [tuple[int, int], tuple[int, int]], tuple[int, int]] = default_serial_to_lattice_index):
        self.contents = rod_contents or {}
        self.lattice = HexagonalLattice(center, shape, pitch, height, outer_radius, mixture)
        self.prefix_indexing = prefix_indexing

        if suffix_indexing is None:
            self.suffix_indexing = list(map(lambda num: str(num), list(range(1, shape[1] + 1))))
        else:
            self.suffix_indexing = suffix_indexing
        self._site_name_partition = site_name_partition
        self._serial_to_lattice_index = serial_to_lattice_index


    @property
    def lattices(self) -> tuple[HexagonalLattice]:
        return self.lattice,

    lattices.__doc__ = Grid.lattices.__doc__

    def site_index(self, site: Site) -> tuple[HexagonalLattice, tuple[int, int]]:
        divided_site_name = self._site_name_partition(site)
        serial_index = (
            self.prefix_indexing.index(divided_site_name[0]), self.suffix_indexing.index(divided_site_name[1]))
        site_pos = self._serial_to_lattice_index(serial_index, self.lattice.shape)

        return self.lattice, site_pos

    site_index.__doc__ = Grid.site_index.__doc__

    def __repr__(self):
        return f"HexagonalGrid<Lattice: {self.lattice}>"

    def __len__(self) -> int:
        return len(self.contents)

    def sites(self) -> Iterable[Site]:
        return _sites(self.lattice.shape)

    def keys(self) -> Iterable[Site]:
        return self.contents.keys()

    keys.__doc__ = dict.keys.__doc__

    def values(self) -> Iterable[Element]:
        return self.contents.values()

    values.__doc__ = dict.values.__doc__

    def items(self) -> Iterable[tuple[Site, Element]]:
        yield from self.contents.items()

    items.__doc__ = dict.items.__doc__

    def __getitem__(self, key: Site) -> Element:
        return self.contents[key]

    def __delitem__(self, key: Site):
        del self.contents[key]

    def __contains__(self, site: Site):
        return site in self.contents

    def __setitem__(self, key: Site, value: Element) -> None:
        self.contents[key] = value

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all((self.lattice == other.lattice,
                        self.contents == other.contents))
        return False


class SpacedGrid(Grid):
    """
    A grid that has 4 cartesian lattices of the same dimensions symmetric around
    the center with spaces between the lattices.

    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    def __init__(self, center: tuple[cm, cm, cm] | np.ndarray,
                 shape: tuple[int, int],
                 lattice_dimensions: tuple[cm, cm],
                 height: cm,
                 space_dx: cm, space_dy: cm,
                 mixture: Mixture,
                 rod_contents: dict[Site, Element] | None = None,
                 ):
        self._shape = (shape[0] // 2, shape[1] // 2)
        self.space_dx = space_dx
        self.space_dy = space_dy
        self.contents = rod_contents or {}
        self._lattices = tuple(
            CartesianLattice(
                np.array([sx * (space_dx + lattice_dimensions[0] * self._shape[0]) / 2,
                          sy * (space_dy + lattice_dimensions[1] * self._shape[1]) / 2,
                          0]) + np.asarray(center),
                self._shape,
                lattice_dimensions,
                height,
                mixture
            ) for sy, sx in product([-1, 1], [-1, 1])
        )

    def __len__(self): return len(self.contents)

    @property
    def lattices(self) -> tuple[CartesianLattice, ...]:
        return self._lattices

    lattices.__doc__ = Grid.__doc__

    def site_index(self, site: Site) -> tuple[CartesianLattice, tuple[int, int]]:
        i, j = alphabet.index(site[0]), int(site[1:]) - 1
        match (i < self._shape[1], j < self._shape[0]):
            case (True, True):
                return self._lattices[0], (i, j)
            case (True, False):
                return self._lattices[1], (i, j - self._shape[0])
            case (False, True):
                return self._lattices[2], (i - self._shape[1], j)
            case (False, False):
                return self._lattices[3], (
                    i - self._shape[1], j - self._shape[0])

    site_index.__doc__ = Grid.__doc__

    def __repr__(self):
        return f"SpacedGrid<Lattice: {self._lattices[0]}, " \
               f"Lattice: {self._lattices[1]}, " \
               f"Lattice: {self._lattices[2]}, " \
               f"Lattice: {self._lattices[3]}>"

    __getitem__ = CartesianGrid.__getitem__
    __setitem__ = CartesianGrid.__setitem__
    __delitem__ = CartesianGrid.__delitem__
    __contains__ = CartesianGrid.__contains__

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(
                (set(self.lattices) == set(other.lattices),
                 self._shape == other._shape,
                 self.contents == other.contents,
                 self.space_dx == other.space_dx,
                 self.space_dy == other.space_dy))
        return False

    def sites(self) -> Iterable[Site]:
        shape = tuple(2 * np.asarray(self._shape))
        return _sites(shape)

    keys = CartesianGrid.keys
    values = CartesianGrid.values
    items = CartesianGrid.items
    lattice_of = Grid.lattice_of


class GeneralSpacedGrid(Grid):
    """
    This class represents a cartesian grid with holes for special components.
    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    def __init__(self, center: tuple[cm, cm, cm] | np.ndarray,
                 shape: tuple[int, int],
                 lattice_dimensions: tuple[cm, cm],
                 height: cm, holes_x: Sequence[int],
                 holes_y: Sequence[int],
                 spaces_dx: Sequence[cm], spaces_dy: Sequence[cm],
                 mixture: Mixture,
                 rod_contents: dict[Site, Element] | None = None,
                 ):
        """
        Parameters
        ----------

        center: tuple[cm, cm, cm]
         The center of the grid
        shape: tuple[int, int]
         The shape of the grid
        lattice_dimensions: tuple[cm, cm]
         The dimensions of a single lattice box.
        height: cm
         The height of the grid
        holes_x: Sequence[int]
         The indices of the columns immediately after the holes in the grid.
        holes_y: Sequence[int]
         The indices of the rows immediately after the holes in the grid.
        spaces_dx: Sequence[cm]
         The lengths of the holes in the columns.
        spaces_dy: Sequence[cm]
         The lengths of the holes in the rows.
        mixture: Mixture
         The mixture of the empty grid.
        """
        assert len(spaces_dx) == len(holes_x)
        assert len(spaces_dy) == len(holes_y)
        self.shape = shape
        self.holes_x = holes_x
        self.holes_y = holes_y
        self.spaces_dx = spaces_dx
        self.spaces_dy = spaces_dy
        self.contents = rod_contents or {}
        self._lattices = tuple(
            CartesianLattice(
                np.array([x_dc + (x0 + x1) / 2 * lattice_dimensions[0] - sum(spaces_dx) / 2 - lattice_dimensions[0] *
                          shape[0] / 2,
                          y_dc + (y0 + y1) / 2 * lattice_dimensions[1] - sum(spaces_dy) / 2 - lattice_dimensions[1] *
                          shape[1] / 2,
                          0]) + np.asarray(center), (int(x1 - x0), int(y1 - y0)),
                lattice_dimensions,
                height,
                mixture
            ) for ((y0, y1), y_dc), ((x0, x1), x_dc) in
            product(zip(pairwise(prepend(0, append(holes_y, shape[1]))), prepend(0, np.cumsum(spaces_dy))),
                    zip(pairwise(prepend(0, append(holes_x, shape[0]))), prepend(0, np.cumsum(spaces_dx)))))

    @property
    def lattices(self) -> tuple[CartesianLattice, ...]:
        return self._lattices

    lattices.__doc__ = Grid.__doc__

    def site_index(self, site: Site) -> tuple[CartesianLattice, tuple[int, int]]:
        i, j = alphabet.index(site[0]), int(site[1:]) - 1
        for l, ((y0, y1), (x0, x1)) in enumerate(
                product(pairwise(prepend(0, append(self.holes_y, self.shape[1]))),
                        pairwise(prepend(0, append(self.holes_x, self.shape[0]))))):
            if y0 <= i < y1 and x0 <= j < x1:
                return self._lattices[l], (i - y0, j - x0)

    site_index.__doc__ = Grid.__doc__

    def __repr__(self):
        return 'SpacedGrid<Lattice: ' + (', Lattice: '.join(map(str, self._lattices))) + '>'

    __getitem__ = CartesianGrid.__getitem__
    __setitem__ = CartesianGrid.__setitem__
    __delitem__ = CartesianGrid.__delitem__
    __contains__ = CartesianGrid.__contains__

    def sites(self) -> Iterable[Site]:
        return _sites(self.shape)

    def __len__(self) -> int:
        return len(self.contents)

    keys = CartesianGrid.keys
    values = CartesianGrid.values
    items = CartesianGrid.items
    lattice_of = Grid.lattice_of
