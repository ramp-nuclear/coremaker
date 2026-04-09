"""Demonstration of CVL with a SpacedGrid core.

This example builds a 6x6 SpacedGrid (four 3x3 quadrant lattices with
visible gaps between them) and shows:

1. Rod map coloured by type_key
2. Power heatmap
3. Transition map with two legends (core rods + load rods)

Run with::

    python -m coremaker.visualization.examples.example_spaced_grid

"""

from copy import deepcopy
from itertools import product
from pathlib import PurePath
from string import ascii_uppercase

import numpy as np
from coreoperator.mobilization import CyclicShuffle, LoadChain, LoadSite, Remove, Scheme

from coremaker.core import Core
from coremaker.geometries.box import Box
from coremaker.geometries.infinite import infiniteGeometry
from coremaker.grids import SpacedGrid
from coremaker.materials import Mixture
from coremaker.materials.aluminium import aluminium
from coremaker.materials.util import room_temperature
from coremaker.materials.water import make_heavy_water
from coremaker.tree import Node, Tree
from coremaker.visualization import (
    plot_heatmap,
    plot_rod_map,
    plot_scheme,
)
from coremaker.visualization.coregeometry import all_site_geometries, occupied_sites

alphabet = ascii_uppercase

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
heavy_water = make_heavy_water(temp=room_temperature)

# Simple uranium fuel mixture
uranium_fuel = Mixture.by_weight_density({__import__("isotopes").U: 19.1}, room_temperature)

# ---------------------------------------------------------------------------
# Rod templates
# ---------------------------------------------------------------------------
site_size = 8.0
block_size = 6.0
block_height = 60.0
rod_height = 80.0


def _make_rod(type_key: str) -> Tree:
    """Build a fuel-rod tree with the given type_key."""
    coolant_path = PurePath("coolant")
    block_path = coolant_path / PurePath("block")
    fuel_path = block_path / PurePath("fuel")

    coolant_node = Node(infiniteGeometry, mixture=heavy_water)
    block_node = Node(
        geometry=Box((0.0,) * 3, dimensions=(block_size, block_size, rod_height)),
        mixture=aluminium,
    )
    fuel_node = Node(
        geometry=Box((0.0,) * 3, dimensions=(block_size * 0.75, block_size * 0.75, block_height)),
        mixture=uranium_fuel,
    )

    tree = Tree(type_key=type_key)
    tree.nodes.update({coolant_path: coolant_node, block_path: block_node, fuel_path: fuel_node})
    tree.exclusive[coolant_path] = [(block_path, block_node)]
    tree.exclusive[block_path] = [(fuel_path, fuel_node)]
    return tree


fuel_a = _make_rod("fuel_A")
fuel_b = _make_rod("fuel_B")

# ---------------------------------------------------------------------------
# Build a SpacedGrid core (6x6 = four 3x3 quadrants with gaps)
# ---------------------------------------------------------------------------
grid_shape = (6, 6)
space_dx = 4.0  # visible horizontal gap (cm)
space_dy = 4.0  # visible vertical gap (cm)

# Populate: fuel_A in the two left quadrants, fuel_B in the two right quadrants
rod_contents = {}
for letter, col in product(alphabet[: grid_shape[0]], range(1, grid_shape[1] + 1)):
    site = f"{letter}{col}"
    rod_contents[site] = deepcopy(fuel_a if col <= grid_shape[1] // 2 else fuel_b)

spaced_grid = SpacedGrid(
    center=(0.0, 0.0, 0.0),
    shape=grid_shape,
    lattice_dimensions=(site_size, site_size),
    height=rod_height,
    space_dx=space_dx,
    space_dy=space_dy,
    mixture=aluminium,
    rod_contents=rod_contents,
)

# Wrap in a Core — every lattice in the SpacedGrid must appear as a node
pool_dims = (
    grid_shape[0] * site_size + space_dx + 20,
    grid_shape[1] * site_size + space_dy + 20,
    rod_height * 2,
)
pool_geometry = Box((0.0,) * 3, pool_dims)
pool_node = Node(geometry=pool_geometry, mixture=heavy_water)

core_tree = Tree(type_key="spaced_core")
pool_path = PurePath("pool")
core_tree.nodes[pool_path] = pool_node
core_tree.exclusive[pool_path] = []

for i, lat in enumerate(spaced_grid.lattices):
    lat_path = pool_path / PurePath(f"lattice_{i}")
    core_tree.nodes[lat_path] = lat
    core_tree.exclusive[pool_path].append((lat_path, lat))

spaced_core = Core(
    grid=spaced_grid,
    aliases={},
    tree=core_tree,
    outer_geometry=pool_geometry,
)


# ---------------------------------------------------------------------------
# Rod factories for the transition plan
# ---------------------------------------------------------------------------
def fresh_fuel():
    return deepcopy(_make_rod("fresh_fuel"))


def mox_rod():
    return deepcopy(_make_rod("MOX"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    core = spaced_core
    geometries = all_site_geometries(core)
    occupied = occupied_sites(core)
    print(f"SpacedGrid core: {len(list(core.grid.sites()))} sites, {len(occupied)} occupied")

    rng = np.random.default_rng(7)

    # --- 1. Rod map ---
    fig1, ax1 = plot_rod_map(core, title="Rod Types -- SpacedGrid Core")
    fig1.savefig("spaced_rod_map.png", dpi=150, bbox_inches="tight")
    print("Saved: spaced_rod_map.png")

    # --- 2. Power heatmap ---
    site_values = {}
    for site in occupied:
        geom = geometries[site]
        r = np.sqrt(geom.center_x**2 + geom.center_y**2)
        site_values[site] = max(0.3, 1.6 - r / 40.0 + rng.uniform(-0.1, 0.1))

    fig2, ax2 = plot_heatmap(
        core,
        site_values=site_values,
        cmap="YlOrRd",
        units="Relative Power Fraction",
        title="Synthetic Power Map -- SpacedGrid Core",
    )
    fig2.savefig("spaced_power_map.png", dpi=150, bbox_inches="tight")
    print("Saved: spaced_power_map.png")

    # --- 3. Transition map with two legends ---
    scheme = Scheme(actions=(
        CyclicShuffle(["B2", "E5"]),
        CyclicShuffle(["C3", "D4"]),
        LoadSite(fresh_fuel, "A1"),
        LoadChain(fresh_fuel, ["F6", "F3"]),
        LoadSite(mox_rod, "A4"),
        Remove(["C6"]),
    ))

    fig3, ax3 = plot_scheme(
        core,
        scheme,
        title="Transition Map -- SpacedGrid Core",
        rod_color_dict={"fuel_A": "#4a90d9", "fuel_B": "#d97b4a"},
        load_color_dict={"fresh_fuel": "#2ecc71", "MOX": "#e74c3c"},
    )
    fig3.savefig("spaced_transition_map.png", dpi=150, bbox_inches="tight")
    print("Saved: spaced_transition_map.png")


if __name__ == "__main__":
    main()
