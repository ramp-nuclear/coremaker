"""Setting up a core made up of a lattice of fuel rods inside a pool of heavy water.
The core has 4 hafnium control rods.
"""

from copy import deepcopy
from functools import partial
from itertools import product
from operator import truediv
from pathlib import PurePath
from string import ascii_uppercase

import isotopes
import numpy as np
from cytoolz import valmap

from coremaker.core import TREE_NAME, Core
from coremaker.geometries.box import Box
from coremaker.geometries.infinite import infiniteGeometry
from coremaker.grids import CartesianGrid
from coremaker.materials import Mixture
from coremaker.materials.absorbers import hafnium
from coremaker.materials.aluminium import aluminium
from coremaker.materials.util import room_temperature
from coremaker.materials.water import make_heavy_water
from coremaker.transform import Transform
from coremaker.tree import Node, Tree

site_size = 10.0
grid_height = 100.0
lattice_shape = (9, 9)
lattice_limits = (lattice_shape[0] * site_size, lattice_shape[1] * site_size)
alphabet = ascii_uppercase

# Constructing the coolant.
heavy_water = make_heavy_water(temp=room_temperature)
heavy_water_coolant_node = Node(infiniteGeometry, mixture=heavy_water)
coolant_path = PurePath("coolant")

# constructing a fuel rod which consists of an uranium block of size 6 x 6 x 60
# inside an aluminum block of size 8 x 8 x 80.
# setting up the aluminum block
aluminum_block_size = 8.0
aluminum_block_height = 80.0
aluminum_block_dimensions = (*([aluminum_block_size] * 2), aluminum_block_height)
# noinspection PyTypeChecker
aluminum_block_node = Node(geometry=Box((0.0,) * 3, dimensions=aluminum_block_dimensions), mixture=aluminium)
aluminum_block_path = coolant_path / PurePath("aluminum_block")
# setting up the uranium block
U_block_size = 6.0
U_block_height = 60.0
U_block_dimensions = (*([U_block_size] * 2), U_block_height)
# density taken from wikipedia.
U_mixture = Mixture.by_weight_density({isotopes.U: 19.1}, room_temperature)
# noinspection PyTypeChecker
U_block_node = Node(geometry=Box((0.0,) * 3, dimensions=U_block_dimensions), mixture=U_mixture)
U_block_path = aluminum_block_path / PurePath("uranium_block")
# Constructing the tree that represents the fuel rod
fuel_rod_tree = Tree(name="fuel_rod")
fuel_rod_tree.nodes.update(
    {coolant_path: heavy_water_coolant_node, aluminum_block_path: aluminum_block_node, U_block_path: U_block_node}
)
# The aluminum block is an exclusive child of the heavy water
fuel_rod_tree.exclusive[coolant_path] = [(aluminum_block_path, aluminum_block_node)]
# The U block is an exclusive child of the aluminum block.
fuel_rod_tree.exclusive[aluminum_block_path] = [(U_block_path, U_block_node)]

# constructing the aluminum lattice of blocks.

aluminum_grid = CartesianGrid(
    center=(0.0,) * 3,
    shape=lattice_shape,
    dimensions=(site_size,) * 2,
    height=grid_height,
    mixture=aluminium,
    rod_contents={
        f"{letter}{index}": deepcopy(fuel_rod_tree)
        for letter, index in product(alphabet[: lattice_shape[0]], range(1, lattice_shape[1] + 1))
    },
)
aluminum_lattice_name = PurePath("aluminum_lattice")

# constructing the 4 hafnium control rods.
hafnium_block_height = aluminum_block_height
hafnium_block_size = aluminum_block_size
hafnium_block_dimensions = (*([hafnium_block_size] * 2), hafnium_block_height)
hafnium_transforms = {
    "south": Transform(translation=np.array([0, -1, 0]) * lattice_limits[1] * 3 / 4),
    "east": Transform(translation=np.array([1, 0, 0]) * lattice_limits[0] * 3 / 4),
    "north": Transform(translation=np.array([0, 1, 0]) * lattice_limits[1] * 3 / 4),
    "west": Transform(translation=np.array([-1, 0, 0]) * lattice_limits[0] * 3 / 4),
}
hafnium_block_names = {direction: PurePath(f"{direction}_hafnium_block") for direction in hafnium_transforms.keys()}
# noinspection PyTypeChecker
hafnium_block_nodes = {
    hafnium_block_names[direction]: Node(
        Box((0.0,) * 3, hafnium_block_dimensions, transform=transform), mixture=hafnium
    )
    for direction, transform in hafnium_transforms.items()
}

# Setting up the core.
core_tree = Tree(name="core")
# Adding the nodes
heavy_water_pool_dimensions = tuple(np.array([*lattice_limits, grid_height]) * 2)
heavy_water_pool_geometry = Box((0.0,) * 3, heavy_water_pool_dimensions)
heavy_water_pool_node = Node(geometry=heavy_water_pool_geometry, mixture=heavy_water)
heavy_water_pool_name = PurePath("pool")
core_tree.nodes[heavy_water_pool_name] = heavy_water_pool_node
aluminum_lattice_path = heavy_water_pool_name / aluminum_lattice_name
core_tree.nodes[aluminum_lattice_path] = aluminum_grid.lattice
hafnium_block_paths = valmap(partial(truediv, heavy_water_pool_name), hafnium_block_names)
core_tree.nodes.update(zip(hafnium_block_paths.values(), hafnium_block_nodes.values()))
# adding the exclusion relations.
core_tree.exclusive = {heavy_water_pool_name: [(aluminum_lattice_path, aluminum_grid.lattice)]}
core_tree.exclusive[heavy_water_pool_name] += [*zip(hafnium_block_paths.values(), hafnium_block_nodes.values())]
hafnium_block_aliases = {
    direction: (f"The {direction}ern hafnium block", (TREE_NAME / path,))
    for direction, path in hafnium_block_paths.items()
}
hafnium_block_aliases.update(
    {"All": ("All the hafnium blocks", tuple(TREE_NAME / path for path in hafnium_block_paths.values()))}
)
example_core = Core(
    grid=aluminum_grid, aliases=hafnium_block_aliases, tree=core_tree, outer_geometry=heavy_water_pool_geometry
)
