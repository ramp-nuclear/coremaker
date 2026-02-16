"""Package for the creation and definition of core objects.

"""

from coremaker.core import Core
from coremaker.geometries import jsonable as geom_json
from coremaker.geometries.holed import ConcreteHoledGeometry as _ChG
from coremaker.grids import jsonable as grid_json
from coremaker.materials import Mixture
from coremaker.mesh import jsonable as mesh_json
from coremaker.protocols.node import NodeLike
from coremaker.surfaces import jsonable as surf_json
from coremaker.tree import Tree, Node

__all__ = ["jsonable", "Mixture"]
jsonable = set([*surf_json, *geom_json, _ChG, Mixture, Tree, Node, *grid_json, 
                Core, *mesh_json])

