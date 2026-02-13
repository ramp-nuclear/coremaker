"""Definition of elements using a tree structure.

Trees are made up of nodes, which have progeny. A node represents a logical or
physical piece of an object. For example, a plate in MTR fuel would be a Node,
for it is a logical piece of the greater MTR fuel assembly. The fuel meat could
also be a Node, which is related to the plate since if the plate moves, so does
the fuel meat.
In this thematic encapsulation, a node always has an external geometry, i.e.
the spatial shape where you can find the things we talk about, as well as a
:class:`coremaker.transform.Transform`, which signifies where the object is located
and what its orientation is. Since it makes more sense to build things relative
to one another, the Transform of each node is relative to its immediate parent.

In this tree implementation we follow three types of children:

* Inclusive children
* Exclusive children
* Externally Exclusive children

Inclusive children are the pieces the node is made up of. For example, if you
split the fuel meat of an MTR fuel plate into X-Y-Z chunks, the meat node would
have each chunk as an Inclusive child, since together the meat node chunks make
up the full meat.

Exclusive children are internal objects that lie within (and thus are excluded from)
the region of an encapsulating object. For example, in the same MTR fuel plate, the fuel cladding
has the outer geometry as defined by the fuel specification, but internally the
cladding could be thicker or thinner depending on the exact shape of the fuel meat.
Thus, the fuel plate is defined as a node with one exclusive child - the meat
(which is also a node).

An externally exclusive child is an exclusive child that does not have to lie
within the region of its parent. Among the three types,
Externally Exclusive children are the most elusive. Sometimes, especially when
the node we consider is made up of a fluid, we could insert something into this
node while movement of the fluid would not change the location of that other object.
For example, if you have a retractable radiation channel than can move into a
known spatial chunk of water, we would like the water to be defined where the
radiation channel is not, but we don't want movement of the water to move the channel,
or vice-versa, and the radiation channel could move as part of a larger object.
In this case, we would mark the radiation channel as an Externally Exclusive
child of the water chunk, so that adapters know they should place the water only
where the channel is not, while making it easier to not include this relationship
when figuring out Transforms and the like.

You should only use Externally Exclusive children when you have to, since they
are tricky and an advanced concept. Most things can be done with regular Inclusive
and Exclusive progeny.

Notice that any node that has internal children is materially defined by those
children, while any node with no internal children must have a material description
for us to know what exists inside that geometry. Therefore, a node should have
a mixture associated with it if and only if it has no internal progeny.
Otherwise, the tree is considered Illegal.

To understand the concept, it's easier if you look at an example of a tree
creation in any of the factories under :ref:`Elements`.

"""
from enum import Enum, auto
from itertools import product
from pathlib import PurePath
from typing import Optional, Tuple, Dict, List, Iterable

from coremaker.component import ConcreteComponent
from coremaker.geometries.holed import ConcreteHoledGeometry
from coremaker.geometries.infinite import infiniteGeometry
from coremaker.geometries.union import ConcreteUnionGeometry
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.grid import Lattice
from coremaker.protocols.mixture import Mixture
from coremaker.protocols.node import NodeLike
from coremaker.transform import Transform, identity


class Node:
    """A node is a structure in the tree.

    """

    def __init__(self, geometry: Geometry,
                 transform: Transform = identity,
                 mixture: Mixture | None = None):
        """
        Parameters
        ----------
        geometry: Geometry
            The external geometry of the node.
        transform: Transform
            The transformation of the node and its progeny relative to its parent.
        mixture: Optional[Mixture]
            The material the node is made out of. Exists iff it has no internal nodes,
            when legal.
        """
        self._geometry = geometry
        self.transform = transform
        self.mixture = mixture

    @property
    def geometry(self) -> Geometry:
        """Gets the geometry of the node.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: Geometry) -> None:
        self._geometry = geometry

    def __eq__(self, other: NodeLike) -> bool:
        return (self.geometry == other.geometry
                and self.mixture == other.mixture
                and self.transform == other.transform)

    def __repr__(self) -> str:
        return f'Node<geometry: {self.geometry}, ' \
               f'mixture: {self.mixture}, ' \
               f'transform: {self.transform}>'


PathNode = Tuple[PurePath, NodeLike]
PathComp = Tuple[PurePath, ConcreteComponent]
Progeny = List[PathNode]


def _condition(p: PurePath, root: PurePath) -> bool:
    return root == p or root in p.parents


def _switch(p: PurePath, old: PurePath, new: PurePath) -> PurePath:
    return new / p.relative_to(old) if old in p.parents or old == p else p


class ChildType(Enum):
    """Relation types between parent and offspring nodes.
    See the documentation for :class:`Tree`

    """
    inclusive = auto()
    exclusive = auto()
    external_exclusive = auto()


class Tree:
    """A tree object. These objects define complex objects by defining
    geometries within geometries and the exclusion rules thereof.

    inclusive relationship means that the parent is made out of several other pieces,
    which, when combined, form the full geometry.
    A good example is when a box is split into multiple cells. The full box is made up
    of those cells, and only together do they have the full external geometry.

    exclusive means that the parent object fully includes inside it other objects,
    which are fully connected to it, but the outer object is different from its
    children. A good example is a box in a frame. The frame is made out of one thing,
    the box of another, and if the frame moves, it takes the box along with it.

    external_exclusive means that the parent object is defined where another
    unconnected object doesn't already take space. This is important for example
    for fluids. A bath of liquid, when a stone is thrown into it, is defined as
    the fluid bath, but excluding whereever the stone is. Moving water would
    not move the stone by necessity.

    """

    def __init__(self):
        self.nodes: Dict[PurePath, NodeLike] = {}
        self.inclusive: Dict[PurePath, Progeny] = {}
        self.exclusive: Dict[PurePath, Progeny] = {}
        self.external_exclusive: Dict[PurePath, Progeny] = {}

    def __getitem__(self, item: PurePath) -> NodeLike:
        """Returns a node using its path.

        """
        return self.nodes[item]

    def __contains__(self, item: PurePath | NodeLike) -> bool:
        if isinstance(item, PurePath):
            return item in self.nodes
        else:
            return item in self.nodes.values()

    def __eq__(self, other: "Tree") -> bool:
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in
                   ['nodes', 'inclusive', 'exclusive', 'external_exclusive'])

    def lookup(self, node: NodeLike) -> PurePath:
        """Look up the name of specific node in the tree by its object.

        Try to avoid this and to use the paths to objects directly, because
        this is O(n) while those are O(1).

        Parameters
        ----------
        node: NodeLike
            The node to look for.

        Returns
        -------
        PurePath
            The path to the node, if found.

        Raises
        ------
        KeyError
            if unfound.

        """
        for path, n in self.nodes.items():
            if n is node:
                return path
        else:
            raise KeyError("The requested node was not found in the tree")

    def roots(self) -> Iterable[PathNode]:
        """Yields the nodes that have no parent.

        """
        yield from ((path, node) for path, node in self.nodes.items()
                    if len(path.parents) == 1)

    def rename(self, oldpath: PurePath, newpath: PurePath) -> None:
        """Rename a path to a node.
        This is a side effect method. Please only use this during constructions.

        Parameters
        ----------
        oldpath: PurePath
            Old path name.
        newpath: PurePath
            New path name.

        """
        if newpath in self.nodes:
            raise ValueError(
                f"The new path {newpath} already exists in the tree,"
                f"so {oldpath} cannot be renamed to it.")
        if oldpath not in self.nodes:
            raise KeyError(f"Could not find the path {oldpath} to rename it.")
        self.nodes = {_switch(p, oldpath, newpath): node
                      for p, node in self.nodes.items()}
        for attr in ['inclusive', 'exclusive', 'external_exclusive']:
            self.__setattr__(attr, {
                _switch(p, oldpath, newpath): [
                    (_switch(p2, oldpath, newpath), n)
                    for p2, n in lst]
                for p, lst in getattr(self, attr).items()})

    def is_legal(self) -> bool:
        """Returns True iff all nodes either have a mixture or inclusive children.

        To make a tree legal, all nodes must either have inclusive children or a
        viable mixture, or there will be a "hole" in the object.
        Inclusive children cannot exist to a Node that does have a viable mixture,
        or it will be double booked.

        """
        return all(bool(node.mixture) ^ bool(self.inclusive[path])
                   for path, node in self.nodes.items())

    def get_transform(self, path: PurePath,
                      memdict: Optional[Dict[PurePath, Transform]] = None
                      ) -> Transform:
        """Get the recursed transform of going through the tree up to path.

        Parameters
        ----------
        path: PurePath
            The path of the node we get the full transform for.
        memdict: Dict[PurePath, Transform] or None
            Memoization trick.

        Returns
        -------
        Transform
            The transform object for fully recursing all transforms down to path.

        """
        memdict = memdict or {}
        if path in memdict:
            return memdict[path]
        if path == PurePath('.'):
            memdict[path] = identity
            return identity
        value = self.get_transform(path.parent, memdict) @ self.nodes[path].transform
        memdict[path] = value
        return value

    def geometry_of(self, node: PurePath) -> Geometry:
        """Gets the geometry at the given path.

        The geometry is transformed with the absolute path from the root.

        Parameters
        ----------
        node: PurePath
            The path to the node for which we want the geometry.

        """
        return self.nodes[node].geometry.transform(self.get_transform(node))

    def transform(self,
                  path: PurePath | None,
                  transform: Transform,
                  relative_to: Transform | None = None,
                  ) -> None:
        r"""Imperatively transform everything under the tree at a given path.

        Parameters
        ----------
        path: PurePath or None
            The Path to the node to transform. If None, transform the entire tree.
        transform: Transform
            The transformation to apply at the node.
        relative_to: Transform | None
            Transform to transform relative to its coordinates. If None, uses the local coordinates at the path.
            Relative transforms follow the formula: :math:`T_r @ T @ T_r^{-1}`, but what we want is that the absolute
            transform would be :math:`(\prod_i{T_i}) @ t = T_r @ T @ T_r^{-1}`, and therefore:

            .. math::
                t = \left(\prod_i{T_i}\right)^{-1} @ T_r @ T @ T_r^{-1}

            For example, if you want to perform the transformation relative to the "lab coordinate system", i.e. the
            one the root of the tree is defined in, you should use the identity transformation.

            .. warning::
                If you want things to be relative to the coordinate system of some node `other`, you have to figure it
                out yourself, and it isn't right to just do :math:`T_r` = self.get_transform(other).

        """
        if relative_to is None:
            t = transform
        else:
            cur = self.get_transform(path.parent) if path else identity
            t = cur.inv() @ relative_to @ transform @ relative_to.inv()
        if path is None:
            for _, root in self.roots():
                root.transform = t @ root.transform
        else:
            self.nodes[path].transform = t @ self.nodes[path].transform

    def named_components(self, transform: Transform = identity) -> Iterable[PathComp]:
        """Yields named components for the use of adapters.

        Assumes the tree is legal.

        """

        memdict = {}
        for path, node in sorted(self.nodes.items(), key=lambda x: x[0]):
            if node.mixture and not isinstance(node, Lattice):
                transgeo = node.geometry.transform(
                    self.get_transform(path, memdict))
                direct_holes = [
                    child.geometry.transform(self.get_transform(cpath, memdict))
                    for cpath, child in self.exclusive.get(path, ())]
                indirect_holes = [
                    child.geometry.transform(self.get_transform(cpath, memdict))
                    for cpath, child in self.external_exclusive.get(path, ())]
                geometry = (
                    ConcreteHoledGeometry(transgeo, direct_holes,
                                          indirect_holes)
                    if bool(direct_holes) or bool(indirect_holes)
                    else transgeo)
                yield path, ConcreteComponent(node.mixture, geometry.transform(transform))

    def components(self) -> Iterable[ConcreteComponent]:
        """Yields components for the use of adapters.

        Assumes the tree is legal.

        Yields
        ------
        :class:`coremaker.component.ConcreteComponent`
            The components in the tree.

        """
        yield from (c for _, c in self.named_components())

    @property
    def outer_geometry(self) -> ConcreteUnionGeometry | Geometry:
        """UnionGeometry or Geometry

        Get the geometry that encompasses this tree structure.
        Notice that this may be larger than the actual object if the object
        has exclusions that are not contained in any of the nodes.

        """
        ogeoms = [node.geometry.transform(node.transform) for _, node in self.roots()]
        match len(ogeoms):
            case 0:
                return infiniteGeometry
            case 1:
                return ogeoms[0]
            case _:
                return ConcreteUnionGeometry(ogeoms)

    def cut(self, path: PurePath) -> None:
        """
        Cuts a node at this path and all its children.
        This is the opposite of the graft operation.

        Parameters
        ----------
        path: PurePath
         The path to cut in, the node in this path is also removed.

        """
        edge_dicts = [self.exclusive, self.inclusive, self.external_exclusive]
        for node in list(self.nodes.keys()):
            if _condition(node, path):
                self.nodes.pop(node)
                for edge_dict in edge_dicts:
                    if node in edge_dict:
                        edge_dict.pop(node)
        for parent, edge_dict in product(path.parents, edge_dicts):
            if parent in edge_dict:
                edges = [(x, y) for (x, y) in edge_dict[parent] if
                         not _condition(x, path)]
                if len(edges) > 0:
                    edge_dict[parent] = edges
                else:
                    edge_dict.pop(parent)

    def graft(self,
              branch: "Tree",
              node: NodeLike | PurePath,
              relation: ChildType,
              ) -> None:
        """Grafts a branch onto this tree at a given node.

        Parameters
        ----------
        branch: Tree
            The branch to graft onto the tree.
        node: NodeLike or PurePath
            The node where the branch goes on the tree. Can be given either with
            the actual node or with its full string name in the tree.
            The full string name is preferred.
        relation: ChildType
            The type of relationship between the branch and the parent node.
            See :class:`ChildType` for further information.

        """

        try:
            header = node if isinstance(node, PurePath) else self.lookup(node)
        except KeyError:  # Means that lookup raised an error.
            header = node
        if header not in self.nodes:  # if the path isn't there or the node lookup failed
            raise KeyError(
                f"Could not graft onto {header} because it wasn't in the tree")
        if any(header / tail in self.nodes for tail in branch.nodes.keys()):
            clashes = [header / tail for tail in branch.nodes.keys()
                       if header / tail in self.nodes]
            raise ValueError("Cannot graft branch onto tree because there are name "
                             f"clashes: {clashes}")
        self.nodes.update({header / tail: v for tail, v in branch.nodes.items()})
        for attr in ['inclusive', 'exclusive', 'external_exclusive']:
            getattr(self, attr).update(
                {header / tail: [(header / path, v) for path, v in lst]
                 for tail, lst in getattr(branch, attr).items()
                 })
        d = {ChildType.inclusive: self.inclusive,
             ChildType.exclusive: self.exclusive,
             ChildType.external_exclusive: self.external_exclusive
             }[relation]
        extension = ((header / tail, n) for tail, n in branch.roots())
        try:
            d[header].extend(extension)
        except KeyError:
            d[header] = list(extension)

    def subtree(self,
                path: PurePath
                ) -> "Tree":
        """Creates the subtree of a given tree at a given node.

        Parameters
        ----------
        path: PurePath
            The node to cut at, inclusive. I.E. this will be the root of the new tree.

        """
        new = Tree()
        short = PurePath(path.name)
        new.nodes = {_switch(p, path, short): node
                     for p, node in self.nodes.items()
                     if _condition(p, path)
                     }
        for attr in ['inclusive', 'exclusive', 'external_exclusive']:
            new.__setattr__(attr,
                            {_switch(p, path, short): [(_switch(p2, path, short), n)
                                                       for p2, n in lst
                                                       if _condition(p2, path)]
                             for p, lst in getattr(self, attr).items()
                             if _condition(p, path)
                             })
        return new
