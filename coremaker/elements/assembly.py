"""Fuels made out of multiple components arranged alongside one another. 

"""
from pathlib import PurePath
from typing import Callable, Iterable

from coremaker.geometries.infinite import infiniteGeometry
from coremaker.materials import Mixture
from coremaker.protocols.geometry import Geometry
from coremaker.transform import Transform, identity
from coremaker.tree import ChildType, Node, Tree

Factory = Callable[[], Tree]
FactorySpec = tuple[Factory, Transform, str]


def singular_root_construction(factories: Iterable[FactorySpec],
                               *,
                               root_mixture: Mixture | None = None,
                               outer_geometry: Geometry = infiniteGeometry,
                               root_path: PurePath,
                               relationship: ChildType,
                               transform: Transform = identity,
                               ) -> Tree:
    """Creates an assembly where there is one root and each of its progeny is
    a tree constructible from a factory.

    Parameters
    ----------
    factories: Iterable[FactorySpec]
        The factory information for each subtree
    root_mixture: Optional[Mixture]
        The mixture to put at the root. Do not use with inclusive construction.
    outer_geometry: Geometry
        The geometry for the root assembly.
    root_path: PurePath
        The name to give the root. Should be a parent-less path.
    relationship: ChildType
        The type of relationship the subtrees have to the root.
    transform: Transform
        The transformation to apply to the root.

    """
    assembly = Tree()
    root = Node(outer_geometry, transform, root_mixture)
    assembly.nodes[root_path] = root
    for x in factories:
        factory, transform, suffix = x
        branch = factory()
        for rpath, node in tuple(branch.roots()):
            branch.transform(rpath, transform)
            if suffix:
                branch.rename(rpath, PurePath(f'{rpath}_{suffix}'))
        assembly.graft(branch, root_path, relationship)
    return assembly
