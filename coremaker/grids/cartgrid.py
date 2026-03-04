"""Concrete implementations for the Grid and Lattice protocols for cartesian geometry."""

import operator
from itertools import product
from string import ascii_uppercase
from typing import Any, Iterable, Type, TypeVar

from ramp_core.serializable import Serializable, deserialize_default

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

from coremaker.grids.lattice import CartesianLattice
from coremaker.grids.util import deserialize_contents, serialize_contents
from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid, Site
from coremaker.protocols.mixture import Mixture
from coremaker.units import cm

alphabet = ascii_uppercase


def cartesian_sites(shape: tuple[int, int]) -> Iterable[Site]:
    """
    The shape is given in the format of:
    (sites in x direction, sites in y direction).

    Parameters
    ----------
    shape: tuple[int, int]
        Number of rows and columns

    """
    rows, columns = shape[::-1]
    return (f"{letter}{number}" for letter, number in product(alphabet[:rows], range(1, columns + 1)))


class CartesianGrid(Grid):
    """A grid that just has one cartesian lattice.

    Rows are named by uppercase letters and columns by numbers.
    A1 corresponds to the southwest corner.
    """

    ser_identifier = "CartGrid"

    def __init__(
        self,
        center: tuple[cm, cm, cm],
        shape: tuple[int, int],
        dimensions: tuple[cm, cm],
        height: cm,
        mixture: Mixture,
        rod_contents: dict[Site, Element] | None = None,
    ):
        self.contents = rod_contents or {}
        self.lattice = CartesianLattice(center, shape, dimensions, height, mixture)

    @classmethod
    def from_lattice(cls: Type[Self], contents: dict[Site, Element], lattice: CartesianLattice) -> Self:
        """Create a Cartesian grid given its lattice and its contents.

        Parameters
        ----------
        contents: dict[Site, Element]
            The elements in different sites of the grid
        lattice: CartesianLattice
            The Cartesian lattice the elements are set in

        """
        obj = cls.__new__(cls)
        obj.contents = contents or {}
        obj.lattice = lattice
        return obj

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {"contents": serialize_contents(self.contents), "lattice": self.lattice.serialize()}

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        contents = deserialize_contents(d["contents"], supported=supported)
        lattice = deserialize_default(d["lattice"], supported=supported, default=CartesianLattice)
        return cls.from_lattice(contents, lattice)

    @property
    def lattices(self) -> tuple[CartesianLattice]:
        return (self.lattice,)

    lattices.__doc__ = Grid.lattices.__doc__

    def site_index(self, site: Site) -> tuple[CartesianLattice, tuple[int, int]]:
        return self.lattice, (alphabet.index(site[0]), int(site[1:]) - 1)

    site_index.__doc__ = Grid.site_index.__doc__

    def __repr__(self):
        return f"CartesianGrid<Lattice: {self.lattice}>"

    def sites(self) -> Iterable[Site]:
        return cartesian_sites(self.lattice.shape)

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
            return all((self.lattice == other.lattice, self.contents == other.contents))
        return NotImplemented

    def __hash__(self):
        items = sorted(self.items(), key=operator.itemgetter(0))
        return hash((self.lattice, tuple(items)))
