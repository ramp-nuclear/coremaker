"""The general protocol other packages should rely on when dealing with a Core.

"""
from pathlib import PurePath
from typing import Protocol, Iterable, Sequence, MutableMapping

from coremaker.protocols.component import Component
from coremaker.protocols.element import Element
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.grid import Grid, Lattice
from coremaker.protocols.node import NodeLike
from coremaker.transform import Transform

Site = str

AliasMap = MutableMapping[str, tuple[str, Sequence[PurePath]]]


class Core(Protocol):
    """The general protocol for how cores should behave in order for packages to use them.
    This object represents a core which is made up of similarly-sized objects in a lattice or in other words, a "Grid".

    Methods are documented where they are. The additional attributes are given here:

    Attributes
    ----------
    aliases: MutableMapping[str, tuple[str, Sequence[PurePath]]]
        Aliases serve to access a logical set of core elements.
        For example, one can use an alias for easy access to the set of elements that make up an absorbing rod inside
        the core.
        Each use case of these aliases is different, and when you set up aliases, you should document what they are for.
        The structure is as follows:
        You put in a key (str), and you get a tuple whose first entry is a string of the explanation for
        what set of elements the alias specifies, and a sequence of keys that serve to access that set of elements.
    grid: Grid
        The grid of the core.

    """

    aliases: AliasMap
    grid: Grid

    def get(self, *keys: PurePath | Site) -> Iterable[NodeLike]:
        """A method to get contents of the core's grid. If you want just one,
        use the __getitem__ instead, since this method always returns an
        iterable.

        Parameters
        ----------
        keys: PurePath
            The keys to look for in the core.

        Returns
        -------
        Iterable[NodeLike]
            A finite iterable of the same size as the keys.

        """
        for key in keys:
            yield self.grid[key]

    def __getitem__(self, key: PurePath | Site) -> NodeLike:
        ...

    def site_transform(self, site: Site) -> Transform:
        """Gets the absolute transform of a grid site.

        This is the transform that must be applied on objects in the site if we want to know where they are in absolute
        terms.

        Parameters
        ----------
        site: Site
            The site to get the transform for.

        """
        raise NotImplementedError("This is not implemented at the protocol level")

    def transform_of(self, path: PurePath) -> Transform:
        """Gets the absolute transform for the given path.

        Parameters
        ----------
        path: PurePath
            The path to the required node.

        """
        raise NotImplementedError("This is not implemented at the protocol level")

    def lattices(self) -> Iterable[tuple[PurePath, Transform, Lattice]]:
        """Yields triplets of lattice name, where it is and the lattice itself.

        """
        raise NotImplementedError("This is not implemented at the protocol level")

    @property
    def free_elements(self) -> Iterable[tuple[str, Element]]:
        """An iterable over the core elements that are not in the grid
        """
        raise NotImplementedError("This is not implemented at the protocol level")

    @property
    def all_elements(self) -> Iterable[Element]:
        """An iterable over all the core elements, in the grid or otherwise.

        """
        yield from (elem for _, elem in self.free_elements)
        yield from self.grid.values()

    @property
    def named_components(self) -> Iterable[tuple[PurePath, Component]]:
        """An iterable over all the components and their paths
        """
        for root, elem in self.grid.items():
            for path, comp in elem.named_components(transform=
                                                    self.site_transform(root)):
                yield PurePath(root) / path, comp
        for root, elem in self.free_elements:
            for path, comp in elem.named_components():
                yield PurePath(root) / path, comp

    @property
    def nodes(self) -> Iterable[tuple[PurePath, NodeLike]]:
        """The nodes that make up the core.
        Use this to perform sweeping modifications on the core.
        __getitem__ or get should be used to perform localized modifications.

        """
        raise NotImplementedError("This is not implemented at the protocol level")

    @property
    def outer_geometry(self) -> Geometry:
        """The outer geometry where everything in the core is bound.
        This is useful for boundary conditions, and for programs that want
        to model things outside the core as well, where the core is just one
        part of the model.

        """
        raise NotImplementedError("This is not implemented at the protocol level")
