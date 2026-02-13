"""A concrete implementation of the Core protocol using a Tree-like object.

After a lengthy discussion, we agreed on the following structure:
The Core object would be similar to the Tree object. The lattices would just
be leaf nodes on that tree, and they would contain the actual lattice-based
elements (which are also, in this case, trees).
This separation where the lattice based rods are not grafted onto the core is
because we don't want to have to rename all the paths on the rods as they move
in the core, and that we want rods to be encapsulated at the core level.

Thus, the Core object is a like a regular tree, except it is legal when all
nodes have one of the following:

#. Inclusive progeny
#. Have a material content
#. Are a leaf node and are a Lattice.

The path to any core component inside a lattice is of the form::

    /path/to/lattice/site/path/in/element

which is just a concatenation of the path to the lattice, the site and the
path within the element at the site to the component.

"""
from pathlib import PurePath
from typing import Optional, Iterable

from coremaker.geometries.union import ConcreteUnionGeometry
from coremaker.protocols.core import Core as CoreProtocol, AliasMap, Site
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.grid import Grid, Lattice
from coremaker.protocols.node import NodeLike
from coremaker.transform import Transform
from coremaker.tree import Tree

TREE_NAME = PurePath('CoreTree')


class Core(CoreProtocol):
    """A concrete implementation of the Core protocol using a Tree-like object.

    """

    def __init__(self, grid: Grid,
                 aliases: AliasMap,
                 tree: Tree,
                 outer_geometry: Optional[Geometry] = None
                 ):
        """Initialization.

        See Also
        --------
        :class:`coremaker.protocols.core.Core`

        Parameters
        ----------
        grid: Grid
            The grid this core houses.
        aliases: AliasMap
            A mapping of alias names to an (explanation, Sequence[path]) tuple.
            The sequence of paths are paths in the core to get to Nodes that
            you want to get at through these names. This can be used to give the
            user a handle to the control systems, for example, in different
            configurations.
        tree: Tree
            The tree structure of the core. The lattices in the grid are expected to be leaf nodes of the tree.
            They must be equivalent to the lattices the grid has.
        outer_geometry: Optional[Geometry]
            An outer geometry to house everything in. If you know that you built
            the ex-grid and grid elements such that they make up the entirety
            of some simple geometry, you could save the downstream codes the
            effort of dealing with such a UnionGeometry by just giving this core
            the simplified outer geometry.
        """
        self.grid = grid
        self.aliases = aliases
        self.tree = tree
        self._outer_geometry = outer_geometry

    def __getitem__(self, key: PurePath) -> NodeLike:
        """Gets you a node using its path.

        Parameters
        ----------
        key: PurePath
            A key is pharsed as a path. The path has the form:
            site/path/in/element if this accesses a site or just path/to/node if it does not.

        Raises
        ------
        KeyError
            if path not in the core

        """
        site = key.parents[-2]
        if site == TREE_NAME:
            stem = key.relative_to(TREE_NAME)
            return self.tree[stem]
        try:
            rod = self.grid[str(site)]
            return rod[key.relative_to(site)]
        except (ValueError, KeyError) as e:
            raise KeyError(f"Could not find {key} in core.") from e

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all(
                (self.grid == other.grid,
                 self.aliases == other.aliases,
                 self.tree == other.tree,
                 self._outer_geometry == other._outer_geometry))
        return False

    def site_transform(self, site: Site) -> Transform:
        lat, indices = self.grid.site_index(site)
        lat_transform = self.tree.get_transform(self.tree.lookup(lat))
        return lat_transform @ Transform(lat.center(indices))

    site_transform.__doc__ = CoreProtocol.site_transform.__doc__

    def transform_of(self, path: PurePath) -> Transform:
        try:
            stem = path.relative_to(TREE_NAME)
            return self.tree.get_transform(stem)
        except ValueError:
            site = str(path.parents[-2])
            stem = path.relative_to(site)
            return self.site_transform(site) @ self.grid[site].get_transform(stem)

    transform_of.__doc__ = CoreProtocol.transform_of.__doc__

    def lattices(self) -> Iterable[tuple[PurePath, Transform, Lattice]]:
        mem = {}
        for lattice in self.grid.lattices:
            path = self.tree.lookup(lattice)
            yield path, self.tree.get_transform(path, memdict=mem), lattice

    lattices.__doc__ = CoreProtocol.lattices.__doc__

    @property
    def free_elements(self) -> Iterable[tuple[str, Tree]]:
        yield str(TREE_NAME), self.tree
    free_elements.__doc__ = CoreProtocol.free_elements.__doc__

    @property
    def nodes(self) -> Iterable[tuple[PurePath, NodeLike]]:
        for path, node in self.tree.nodes.items():
            yield TREE_NAME / path, node
        for site, rod in self.grid.items():
            for path, node in rod.nodes.items():
                yield PurePath(site) / path, node

    nodes.__doc__ = CoreProtocol.nodes.__doc__

    def _root_geometry(self) -> Geometry:
        roots = [n.geometry.transform(self.transform_of(TREE_NAME / p))
                 for p, n in self.tree.roots()]
        return roots[0] if len(roots) == 1 else ConcreteUnionGeometry(roots)

    @property
    def outer_geometry(self) -> Geometry:
        return self._outer_geometry or self._root_geometry()

    outer_geometry.__doc__ = CoreProtocol.outer_geometry.__doc__
