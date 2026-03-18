"""The Protocol that defines what a grid has to do to be a grid."""

from abc import abstractmethod
from typing import Hashable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
from ramp_core.serializable import Serializable

from coremaker.protocols.element import Element
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.mixture import Mixture
from coremaker.protocols.node import NodeLike

__all__ = ["Lattice", "Grid", "Site"]


Site = str


@runtime_checkable
class Lattice(NodeLike, Hashable, Protocol):
    """A lattice is a 3D tiling of the space with some repeating structure.

    For example, the plane can be split into adjoining squares, or a honeycomb of hexagons.
    A parraleloid grid is also possible, as is a triangular one.
    Then, it can be cut parallel to the z-axis in prisms.
    A non-prismatic lattice is possible, for example with tetrahedrons, but this
    is probably not viable in any engineering sense or for most transport codes.

    """

    mixture: Mixture

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the lattice when encoded as a tuple of integers."""
        raise NotImplementedError("This is not supported at the protocol level")

    @property
    def origin(self) -> np.ndarray:
        """The base translation of the lattice relative to its parent.

        Returns
        -------
        An ArrayLike object.

        """
        raise NotImplementedError("This is not supported at the protocol level")

    def center(self, index: tuple[int, int]) -> tuple[float, float, float]:
        """Get the center of a given position in the lattice.

        Parameters
        ----------
        index: tuple[int, int]
            The index in the lattice to get the center of.

        Returns
        -------
        A triplet of floats for a location in 3D space.

        """
        raise NotImplementedError("This is not supported at the protocol level")

    @property
    def inner_geometry(self) -> Geometry:
        """The outer geometry of each tile in the lattice."""
        raise NotImplementedError("This is not supported at the protocol level")


class Grid(Serializable, Protocol):
    r"""A core's grid is where elements are places into the core in a sort of lattice.

    A grid is basically made out of lattices, which are a tiling of a 2D space
    with some repeating shape. Lattices are made up of sites, which are the
    individual tiles in lattice.

    The sites in all the lattices combined are named at the grid level, but
    individual lattices must be able to describe their tiles, as well.
    For example, consider the following 4 lattice grid, which is shaped as the OPAL reactor's grid::

              1 2   3 4
        A     X X | Y Y
        B     X X | Y Y
              ----o----
        C     U U | V V
        D     U U | V V

    The site 'A2' is in the north-west lattice, marked here as X. But inside that
    lattice, it is in the first row and second column. Thus, at the grid level
    its name would be A2, but at the lattice level its name would be (1,2), to
    ensure that they're different enough no one would confuse the two.

    """

    @property
    def lattices(self) -> Sequence[Lattice]:
        """An iterable over the lattices that make up the grid.

        The order of the lattices is lexicographic.
        You take the site that is lexicographically minimal in each lattice, and order the lattices by
        the lexicographic order of those minimal sites.

        """
        raise NotImplementedError("The protocol doesn't support this directly.")

    def site_index(self, site: Site) -> tuple[Lattice, tuple[int, int]]:
        """Get the lattice and the site's index within the lattice for the site.

        Parameters
        ----------
        site: str
            The site name.

        Returns
        -------
        Lattice and internal index.

        """
        raise NotImplementedError("The protocol doesn't support this directly.")

    def lattice_of(self, site: Site) -> Lattice:
        """Get the lattice of a given site.

        Parameters
        ----------
        site: str
            The site for which we want the lattice.

        """
        return self.site_index(site)[0]

    @abstractmethod
    def __getitem__(self, key: Site) -> Element:
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def __setitem__(self, key: Site, value: Element) -> None:
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def __delitem__(self, key):
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def __contains__(self, site: Site) -> bool:
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def sites(self) -> Iterable[Site]:
        """
        Returns an iterable over all available sites, occupied or unoccupied.
        """
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def keys(self) -> Iterable[Site]:
        """Same as a dictionary's keys method. Keys are occupied sites."""
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def values(self) -> Iterable[Element]:
        """Same as a dictionary's values method."""
        raise NotImplementedError("The protocol doesn't support this directly.")

    @abstractmethod
    def items(self) -> Iterable[tuple[str, Element]]:
        """Same as a dictionary's items method."""
        raise NotImplementedError("The protocol doesn't support this directly.")

    def __len__(self):
        return len(list(self.keys()))
