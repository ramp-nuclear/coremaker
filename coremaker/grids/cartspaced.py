from itertools import pairwise, product
from typing import Any, Iterable, Sequence, Type, TypeVar

from ramp_core.serializable import Serializable, deserialize_default

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from more_itertools import prepend
from numpy import append

from coremaker.grids.cartgrid import CartesianGrid, alphabet, cartesian_sites
from coremaker.grids.lattice import CartesianLattice
from coremaker.grids.util import deserialize_contents, serialize_contents
from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid, Site
from coremaker.protocols.mixture import Mixture
from coremaker.units import cm


class SpacedGrid(Grid):
    """
    A grid that has 4 cartesian lattices of the same dimensions symmetric around
    the center with spaces between the lattices.

    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    ser_identifier = "CartSpacedGrid"

    def __init__(
        self,
        center: tuple[cm, cm, cm],
        shape: tuple[int, int],
        lattice_dimensions: tuple[cm, cm],
        height: cm | None,
        space_dx: cm,
        space_dy: cm,
        mixture: Mixture,
        rod_contents: dict[Site, Element] | None = None,
    ):
        self._shape = (shape[0] // 2, shape[1] // 2)
        self.space_dx = space_dx
        self.space_dy = space_dy
        self.contents: dict[Site, Element] = rod_contents or {}
        lat_centers = [
            np.array(
                [
                    sx * (space_dx + lattice_dimensions[0] * self._shape[0]) / 2,
                    sy * (space_dy + lattice_dimensions[1] * self._shape[1]) / 2,
                    0,
                ]
            )
            + np.asarray(center)
            for sy, sx in product([-1, 1], [-1, 1])
        ]
        # Safe because we build the centers to be 3-tuples above
        self._lattices = tuple(
            CartesianLattice(
                tuple(lat_center),  # type: ignore
                self._shape,
                lattice_dimensions,
                height,
                mixture,
            )
            for lat_center in lat_centers
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @classmethod
    def from_lattices(
        cls: Type[Self],
        shape: tuple[int, int],
        space_dx: cm,
        space_dy: cm,
        lattices: tuple[CartesianLattice, CartesianLattice, CartesianLattice, CartesianLattice],
        rod_contents: dict[Site, Element] | None = None,
    ):
        """Create a spaced grid from pre-computed lattices

        Parameters
        ----------
        shape: tuple[int, int]
            The shape of the grid (how many cells by how many cells in each lattice)
        space_dx: cm
            Spacing on the x-axis
        space_dy: cm
            Spacing on the y-axis
        lattices: 4 Cartesian Lattices
            The lattices that make up the grid
        rod_contents: dict[Site, Element]
            Mapping of what already exists in the grid

        """

        obj = cls.__new__(cls)
        obj._shape = shape
        obj.space_dx, obj.space_dy = space_dx, space_dy
        obj.contents = rod_contents or {}
        obj._lattices = lattices
        return obj

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {
            "shape": list(self._shape),
            "dx": self.space_dx,
            "dy": self.space_dy,
            "contents": serialize_contents(self.contents),
            "lattices": [lat.serialize() for lat in self.lattices],
        }

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        shape = tuple(d["shape"])
        contents = deserialize_contents(d["contents"], supported=supported)
        lattices = tuple(deserialize_default(v, supported=supported, default=CartesianLattice) for v in d["lattices"])
        return cls.from_lattices(
            shape=shape, space_dx=d["dx"], space_dy=d["dy"], rod_contents=contents, lattices=lattices
        )

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
                return self._lattices[3], (i - self._shape[1], j - self._shape[0])
            case _:
                raise RuntimeError("There is no way this is reached, cases above should be extensive")

    site_index.__doc__ = Grid.__doc__

    def __repr__(self):
        return (
            f"SpacedGrid<Lattice: {self._lattices[0]}, "
            f"Lattice: {self._lattices[1]}, "
            f"Lattice: {self._lattices[2]}, "
            f"Lattice: {self._lattices[3]}>"
        )

    __getitem__ = CartesianGrid.__getitem__
    __setitem__ = CartesianGrid.__setitem__
    __delitem__ = CartesianGrid.__delitem__
    __contains__ = CartesianGrid.__contains__

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(
                (
                    self.lattices == other.lattices,
                    self._shape == other._shape,
                    self.contents == other.contents,
                    self.space_dx == other.space_dx,
                    self.space_dy == other.space_dy,
                )
            )
        return NotImplemented

    def __hash__(self):
        return hash((self.lattices, self._shape, self.contents, self.space_dx, self.space_dy))

    def sites(self) -> Iterable[Site]:
        x, y = self.shape
        return cartesian_sites((2 * x, 2 * y))

    keys = CartesianGrid.keys
    values = CartesianGrid.values
    items = CartesianGrid.items
    lattice_of = Grid.lattice_of


def _gen_holes(holes: Iterable[int], grid_length: int):
    return prepend(0, append(holes, grid_length))


class GeneralSpacedGrid(Grid):
    """
    This class represents a cartesian grid with holes for special components.
    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    ser_identifier = "GenCartSpacedGrid"

    def __init__(
        self,
        center: tuple[cm, cm, cm],
        shape: tuple[int, int],
        lattice_dimensions: tuple[cm, cm],
        height: cm,
        holes_x: Sequence[int],
        holes_y: Sequence[int],
        spaces_dx: Sequence[cm],
        spaces_dy: Sequence[cm],
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

        spacings_y, spacings_x = tuple(prepend(0, np.cumsum(v)) for v in (spaces_dy, spaces_dx))
        holes_y, holes_x = tuple(
            _gen_holes(holes, grid_length) for holes, grid_length in zip((holes_y, holes_x), (shape[1], shape[0]))
        )

        def _lcenter(y0, y1, y_dc, x0, x1, x_dc) -> tuple[float, float, float]:
            xmean, ymean = (x0 + x1) / 2, (y0 + y1) / 2
            # Safe because we create it as 3-tuples explicitly, we just use vectors incidentally.
            return tuple(
                np.array(
                    [  # type: ignore
                        x_dc
                        + xmean * lattice_dimensions[0]
                        - sum(spaces_dx) / 2
                        - lattice_dimensions[0] * shape[0] / 2,
                        y_dc
                        + ymean * lattice_dimensions[1]
                        - sum(spaces_dy) / 2
                        - lattice_dimensions[1] * shape[1] / 2,
                        0,
                    ]
                )
                + center
            )

        self._lattices = tuple(
            CartesianLattice(
                center=_lcenter(y0, y1, y_dc, x0, x1, x_dc),
                shape=(int(x1 - x0), int(y1 - y0)),
                dimensions=lattice_dimensions,
                height=height,
                mixture=mixture,
            )
            for ((y0, y1), y_dc), ((x0, x1), x_dc) in product(
                zip(pairwise(holes_y), spacings_y), zip(pairwise(holes_x), spacings_x)
            )
        )

    @classmethod
    def from_lattices(
        cls: Type[Self],
        shape: tuple[int, int],
        holes_x: Sequence[int],
        holes_y: Sequence[int],
        spaces_dx: Sequence[cm],
        spaces_dy: Sequence[cm],
        lattices: Sequence[CartesianLattice],
        contents: dict[Site, Element] | None = None,
    ) -> Self:
        """Create a new spaced grid from existing lattice objects.

        Parameters
        ----------
        shape: tuple[int, int]
            The shape of the grid.
        holes_x: Sequence[int]
            The indices of the columns immediately after the holes in the grid.
        holes_y: Sequence[int]
            The indices of the rows immediately after the holes in the grid.
        spaces_dx: Sequence[cm]
            The lengths of the holes in the columns.
        spaces_dy: Sequence[cm]
            The lengths of the holes in the rows.
        lattices: Sequence[CartesianLattice]
            Existing lattice objects.
        contents: dict[Site, Element]
            Existing elements that are already in the grid.

        """
        assert len(spaces_dx) == len(holes_x)
        assert len(spaces_dy) == len(holes_y)
        assert len(lattices) == (len(holes_x) + 1) * (len(holes_y) + 1)
        obj = cls.__new__(cls)
        obj.shape = shape
        obj.holes_x = holes_x
        obj.holes_y = holes_y
        obj.spaces_dx = spaces_dx
        obj.spaces_dy = spaces_dy
        obj.contents = contents or {}
        obj._lattices = tuple(lattices)
        return obj

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, dict(
            shape=list(self.shape),
            holes_x=list(self.holes_x),
            holes_y=list(self.holes_y),
            holes_dx=list(self.spaces_dx),
            holes_dy=list(self.spaces_dy),
            contents=serialize_contents(self.contents),
            lattices=[lat.serialize() for lat in self.lattices],
        )

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        shape = tuple(d["shape"])
        contents = deserialize_contents(d["contents"], supported=supported)
        lattices = [deserialize_default(v, supported=supported, default=CartesianLattice) for v in d["lattices"]]
        return cls.from_lattices(
            shape=shape,
            holes_x=d["holes_x"],
            holes_y=d["holes_y"],
            spaces_dx=d["holes_dx"],
            spaces_dy=d["holes_dy"],
            contents=contents,
            lattices=lattices,
        )

    def __eq__(self, other: "GeneralSpacedGrid"):
        if isinstance(other, GeneralSpacedGrid):
            return all(getattr(self, x) == getattr(other, x) for x in ["shape", "contents", "lattices"]) and all(
                all(y == z for y, z in zip(getattr(self, x), getattr(other, x)))
                and len(getattr(self, x)) == len(getattr(other, x))
                for x in ["holes_x", "holes_y", "spaces_dx", "spaces_dy"]
            )
        return NotImplemented

    def __hash__(self):
        return hash(
            (
                self.shape,
                tuple(self.holes_y),
                tuple(self.holes_x),
                tuple(self.spaces_dy),
                tuple(self.spaces_dx),
                tuple(self.contents.items()),
                self.lattices,
            )
        )

    @property
    def lattices(self) -> tuple[CartesianLattice, ...]:
        return self._lattices

    lattices.__doc__ = Grid.__doc__

    def site_index(self, site: Site) -> tuple[CartesianLattice, tuple[int, int]]:
        i, j = alphabet.index(site[0]), int(site[1:]) - 1
        for lat, ((y0, y1), (x0, x1)) in enumerate(
            product(
                pairwise(prepend(0, append(self.holes_y, self.shape[1]))),
                pairwise(prepend(0, append(self.holes_x, self.shape[0]))),
            )
        ):
            if y0 <= i < y1 and x0 <= j < x1:
                return self._lattices[lat], (i - y0, j - x0)
        else:
            raise KeyError(f"Could not find site index for site {Site}")

    site_index.__doc__ = Grid.__doc__

    def __repr__(self):
        return "SpacedGrid<Lattice: " + (", Lattice: ".join(map(str, self._lattices))) + ">"

    __getitem__ = CartesianGrid.__getitem__
    __setitem__ = CartesianGrid.__setitem__
    __delitem__ = CartesianGrid.__delitem__
    __contains__ = CartesianGrid.__contains__

    def sites(self) -> Iterable[Site]:
        return cartesian_sites(self.shape)

    keys = CartesianGrid.keys
    values = CartesianGrid.values
    items = CartesianGrid.items
    lattice_of = Grid.lattice_of
