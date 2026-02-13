"""A data structure capable of generating named components.
The underlying protocol is used for defining what kind of object adapters
have to XXXX with.

"""
from pathlib import PurePath
from typing import Protocol, Tuple, Iterable

from coremaker.protocols.component import Component
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.node import NodeLike
from coremaker.transform import Transform, identity

PathComp = Tuple[PurePath, Component]


class Element(Protocol):
    """An element is a sub structure of a core.
    It is a protocol for any data structure that adapters should be capable of dealing with.

    """

    nodes: dict[PurePath, NodeLike]

    def named_components(self, transform: Transform = identity) -> Iterable[PathComp]:
        """The main use of an element is to generate components, which are
        pieces of the core that can hopefully be modeled. Most Monte Carlo codes
        like OpenMC support the use of geometries defined in a way that is
        consistent with giving them a bunch of components like these.

        Yields
        ------
        A tuple of the component's structural name and the actual component

        """
        ...

    def components(self) -> Iterable[Component]:
        """Similar to :func:`Element.named_components`, this is a generator for components.
        This is just a use case where naming the components doesn't matter.

        See Also
        --------
        :func:`Element.named_components`

        """
        ...

    def get_transform(self, path: PurePath) -> Transform:
        ...

    def geometry_of(self, node: PurePath) -> Geometry:
        """Returns the geometry at the given path.

        Parameters
        ----------
        node: PurePath
            The node for which we want the geometry

        """
        ...

    def transform(self, node: PurePath | None, transform: Transform) -> None:
        """An imperative method to transform everything in a sub-element according to its path.

        Parameters
        ----------
        node: PurePath or None
            The path to the sub element to transform. If None, the transformation should apply to the full Element.
        transform: Transform
            The transformation to apply

        """
        ...

    @property
    def outer_geometry(self) -> Geometry:
        """The geometry that encompasses the element.
        """
        raise NotImplementedError("Not implemented for general elements.")

    def __getitem__(self, key: PurePath):
        raise NotImplementedError("Not implemented at the protocol level.")
