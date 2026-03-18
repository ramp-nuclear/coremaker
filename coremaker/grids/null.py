from typing import Iterable, Sequence, Type, TypeVar

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid, Lattice, Site


class NullGrid(Grid):
    """
    Empty Grid with no sites
    """

    def __len__(self):
        return 0

    ser_identifier = "NullGrid"
    __slots__ = ()

    def __getitem__(self, key: Site) -> Element:
        raise KeyError("There are no items im a NullGrid")

    def __setitem__(self, key: Site, value: Element) -> None:
        pass

    def __delitem__(self, key: Site) -> None:
        pass

    def __contains__(self, key: Site) -> bool:
        return False

    def __init__(self):
        pass

    def serialize(self) -> tuple[str, dict]:
        return self.ser_identifier, {}

    @classmethod
    def deserialize(cls: Type[Self], *_, **__) -> Self:
        return _null_grid

    def __eq__(self, other: Self) -> bool:
        return isinstance(other, NullGrid) or NotImplemented

    def __hash__(self):
        return 37  # Singleton

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

    def site_index(self, site: Site) -> tuple[Lattice, tuple[int, int]]:
        raise KeyError("There are no sites im a NullGrid")


_null_grid = NullGrid()
