"""End-to-end demonstration of CVL using a coreMaker Core object.

This example uses ``coremaker.example.example_core`` -- a 9x9 CartesianGrid
core with uranium fuel rods inside aluminum blocks in a heavy-water pool --
and demonstrates the full visualization workflow:

1. Power heatmap with peak annotation
2. Burnup / depletion heatmap
3. Rod map coloured by type_key
4. Transition map from a coreoperator Scheme

Run with::

    python -m coremaker.visualization.examples.example_coremaker

"""

from copy import deepcopy

import numpy as np
from coremaker.example import example_core, fuel_rod_tree
from coremaker.visualization import (
    plot_heatmap,
    plot_rod_map,
    plot_scheme,
)
from coremaker.visualization.coregeometry import all_site_geometries, occupied_sites
from coreoperator.mobilization import CyclicShuffle, LoadChain, LoadSite, Remove, Scheme


def fresh_fuel_rod():
    """Factory that produces a fresh fuel rod."""
    return deepcopy(fuel_rod_tree)


def control_rod():
    """Factory that produces a control rod (stub for demonstration)."""
    rod = deepcopy(fuel_rod_tree)
    rod.type_key = "control_rod"
    return rod


def main():
    core = example_core
    geometries = all_site_geometries(core)
    occupied = occupied_sites(core)
    print(f"Core: {len(list(core.grid.sites()))} sites, {len(occupied)} occupied")

    rng = np.random.default_rng(42)

    # --- 1. Power map (synthetic RPF) ---
    site_values = {}
    for site in occupied:
        geom = geometries[site]
        r = np.sqrt(geom.center_x**2 + geom.center_y**2)
        site_values[site] = max(0.3, 1.8 - r / 50.0 + rng.uniform(-0.1, 0.1))

    fig1, ax1 = plot_heatmap(
        core,
        site_values=site_values,
        cmap="YlOrRd",
        units="Relative Power Fraction",
        title=f"Synthetic Power Map -- coreMaker {core.grid.lattice.shape} Core",
    )
    peak = max(site_values, key=site_values.get)
    g = geometries[peak]
    ax1.plot(g.center_x, g.center_y, "*k", markersize=14, zorder=5)
    fig1.savefig("coremaker_power_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_power_map.png")

    # --- 2. Burnup map (synthetic depletion) ---
    dep = {
        site: min(
            0.95,
            0.05 + np.sqrt(geometries[site].center_x**2 + geometries[site].center_y**2) / 80.0
            + rng.uniform(0, 0.05),
        )
        for site in occupied
    }

    fig2, ax2 = plot_heatmap(
        core,
        site_values=dep,
        cmap="YlGnBu",
        units="Fraction Depleted",
        title="Synthetic Burnup -- coreMaker Core",
    )
    fig2.savefig("coremaker_burnup_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_burnup_map.png")

    # --- 3. Rod map (coloured by type_key) ---
    fig3, ax3 = plot_rod_map(core, title="Rod Types -- coreMaker Core")
    fig3.savefig("coremaker_rod_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_rod_map.png")

    # --- 4. Transition map from a Scheme ---
    scheme = Scheme(actions=(
        CyclicShuffle(["E5", "C3", "G7"]),
        LoadChain(fresh_fuel_rod, ["D4", "F6", "H8"]),
        LoadSite(control_rod, "B2"),
        Remove(["A1", "I9"]),
    ))

    fig4, ax4 = plot_scheme(
        core,
        scheme,
        title="Shuffling Scheme -- coreMaker Core",
    )
    fig4.savefig("coremaker_transition_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_transition_map.png")


if __name__ == "__main__":
    main()
