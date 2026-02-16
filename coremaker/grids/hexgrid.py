from typing import Sequence, Callable, Iterable, TypeVar, Type, Any

from ramp_core.serializable import Serializable, deserialize_default

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

from coremaker.grids.lattice import HexagonalLattice
from coremaker.grids.cartgrid import alphabet, cartesian_sites
from coremaker.grids.util import serialize_contents, deserialize_contents
from coremaker.materials.water import make_light_water
from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid, Site
from coremaker.protocols.mixture import Mixture
from coremaker.units import cm


Coords = tuple[int, int]
_default_suffixes = tuple(str(i) for i in range(1, len(alphabet) + 1))


def default_serial_to_lattice_index(site_index: Coords, shape: tuple[int, int]) -> tuple[int, int]:
    site_pos = tuple((site_index[0] - shape[0] // 2, int(site_index[1] - shape[1] // 2)))
    return site_pos


def _default_name_partition(site: Site) -> tuple[str, str]: return site[0], site[1:]


T = TypeVar("T")
S = TypeVar("S")


def _func_from_dict(d: dict[T, S]) -> Callable[[T], S]:
    def _pull_from_d(key: T, *_, **__) -> S:
        return d[key]

    return _pull_from_d


class HexagonalGrid(Grid):
    """A grid that just has one hexagonal lattice.
    Rows are named by uppercase letters and columns by numbers.

    Parameters
    ----------
    center:
        Center of the grid in a 3D space in cm
    shape:
        Shape of the hexagon in the default coordinate system see: default_hexagonal_index_position
    pitch:
        Pitch between sites, in cm
    height:
        Height of the cells in the grid, in cm.
    outer_radius:
        Bounding radius of the grid in cm
    mixture:
        Default mixture inside each site
    rod_contents:
        Rods that are loaded inside the grid at creation time
    prefix_indexing:
        A sequence that maps site's name prefix to an index from 0 to shape[0] - 1
    suffix_indexing:
        A sequence that maps site's name suffix to an index from 0 to shape[0] - 1
    site_name_partition:
        Function that split the site name to prefix and suffix, it is here because not always
        the split is trivial. The default is that a letter + number is how a name is built.
    serial_to_lattice_index:
        Maps between the serial index of the site to the lattice index

    Examples
    --------
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

    ser_identifier = "HexGrid"

    def __init__(self,
                 center: tuple[cm, cm, cm],
                 shape: tuple[int, int],
                 pitch: cm,
                 height: cm,
                 outer_radius: cm,
                 mixture: Mixture,
                 rod_contents: dict[Site, Element] | None = None,
                 prefix_indexing: Sequence = alphabet,
                 suffix_indexing: Sequence = _default_suffixes,
                 site_name_partition: Callable[[Site], Coords] = _default_name_partition,
                 serial_to_lattice_index: Callable[[Coords, Coords], Coords] = default_serial_to_lattice_index
                 ):
        self.contents = rod_contents or {}
        self.lattice = HexagonalLattice(center, shape, pitch, height, outer_radius, mixture)
        self.prefix_indexing = prefix_indexing
        self.suffix_indexing = suffix_indexing
        self._site_name_partition = site_name_partition
        self._serial_to_lattice_index = serial_to_lattice_index

    @classmethod
    def from_lattice(cls: Type[Self],
                     contents: dict[Site, Element],
                     lattice: HexagonalLattice,
                     prefix_indexing: Sequence[str] = alphabet,
                     suffix_indexing: Sequence[str] = _default_suffixes,
                     site_name_partition: Callable[[Site], Coords] = _default_name_partition,
                     serial_to_lattice_index: Callable[[Coords, Coords], Coords] = default_serial_to_lattice_index,
                     ) -> Self:
        """Create a grid from an existing Hexagonal lattice.

        Parameters
        ----------
        contents: dict[Site, Element]
            Rods that are already in the grid
        lattice: HexagonalLattice
            Pre-existing lattice object
        prefix_indexing: Sequence[str]
            Indices in a site name, by order, for first index
        suffix_indexing: Sequence[str]
            Indices in a site name, by order, for second index
        site_name_partition: Callable[[Site], Coords]
            Splits a name into coordinates in the lattice
        serial_to_lattice_index: Callable[[Coords, Coords], Coords]
            Maps between the serial index of the site to the lattice index

        """
        obj = cls.__new__(cls)
        obj.contents = contents
        obj.lattice = lattice
        obj.prefix_indexing = prefix_indexing
        obj.suffix_indexing = suffix_indexing
        obj._site_name_partition = site_name_partition
        obj._serial_to_lattice_index = serial_to_lattice_index
        return obj

    def serialize(self) -> tuple[str, dict[str, Any]]:
        serial_indices = [self._serial_index(site) for site in self.sites()]
        return self.ser_identifier, dict(
            contents=serialize_contents(self.contents),
            lattice=self.lattice.serialize(),
            prefix_indexing=list(self.prefix_indexing),
            suffix_indexing=list(self.suffix_indexing),
            name_partition={site: list(self._site_name_partition(site)) for site in self.sites()},
            serial_index=[[list(index), list(self._serial_to_lattice_index(index, self.lattice.shape))]
                          for index in serial_indices]
        )

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        lattice = deserialize_default(d["lattice"], supported=supported, default=HexagonalLattice)
        name_partition = {site: tuple(result) for site, result in d["name_partition"].items()}
        serial = {tuple(ser_index): tuple(lat_index) for ser_index, lat_index in d["serial_index"]}
        return cls.from_lattice(contents=deserialize_contents(d["contents"], supported=supported),
                                lattice=lattice,
                                prefix_indexing=d["prefix_indexing"],
                                suffix_indexing=d["suffix_indexing"],
                                site_name_partition=_func_from_dict(name_partition),
                                serial_to_lattice_index=_func_from_dict(serial),
                                )

    @property
    def lattices(self) -> tuple[HexagonalLattice]:
        return self.lattice,

    lattices.__doc__ = Grid.lattices.__doc__

    def _serial_index(self, site: Site) -> Coords:
        pre, suf = self._site_name_partition(site)
        return self.prefix_indexing.index(pre), self.suffix_indexing.index(suf)

    def site_index(self, site: Site) -> tuple[HexagonalLattice, tuple[int, int]]:
        site_pos = self._serial_to_lattice_index(self._serial_index(site), self.lattice.shape)
        return self.lattice, site_pos

    site_index.__doc__ = Grid.site_index.__doc__

    def __repr__(self):
        return f"HexagonalGrid<Lattice: {self.lattice}>"

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

    def __contains__(self, site: Site):
        return site in self.contents

    def __setitem__(self, key: Site, value: Element) -> None:
        self.contents[key] = value

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all((self.lattice == other.lattice,
                        self.contents == other.contents,
                        all(self.site_index(site) == other.site_index(site) for site in self.sites())
                        ))
        return NotImplemented
