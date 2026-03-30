"""End-to-end demonstration of CVL using a coreMaker Core object.

This example uses ``coremaker.example.example_core`` -- a 9x9 CartesianGrid
core with uranium fuel rods inside aluminum blocks in a heavy-water pool --
and demonstrates the full visualization workflow:

1. Power heatmap with peak annotation
2. Burnup / depletion heatmap
3. Categorical component-type map
4. Shuffling scheme with movement arrows and discharge markers

Run with::

    python -m coremaker.visualization.examples.example_coremaker

"""

import numpy as np
from coremaker.example import example_core

from coremaker.visualization import (
    plot_burnup_map,
    plot_component_type_map,
    plot_heatmap,
    plot_shuffle_map,
)
from coremaker.visualization._core_geometry import all_site_geometries, occupied_sites


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

    fig2, ax2 = plot_burnup_map(core, dep_map=dep, title="Synthetic Burnup -- coreMaker Core")
    fig2.savefig("coremaker_burnup_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_burnup_map.png")

    # --- 3. Component type map ---
    def type_by_position(site, _element):
        geom = geometries[site]
        r = np.sqrt(geom.center_x**2 + geom.center_y**2)
        if r > 35:
            return "Reflector"
        elif r > 20:
            return "Twice-Burned"
        elif r > 10:
            return "Once-Burned"
        return "Fresh UO2"

    fig3, ax3 = plot_component_type_map(
        core,
        type_fn=type_by_position,
        color_dict={
            "Fresh UO2": "#e41a1c",
            "Once-Burned": "#ff7f00",
            "Twice-Burned": "#4daf4a",
            "Reflector": "#999999",
        },
        title="Assembly Types -- coreMaker Core",
    )
    fig3.savefig("coremaker_type_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_type_map.png")

    # --- 4. Shuffle map with discharges ---
    movements = [
        ("E5", "C3"),
        ("G7", "E5"),
        ("D4", "F6"),
        ("H8", "G7"),
    ]
    discharges = ["A1", "I9", "A9"]

    fig4, ax4 = plot_shuffle_map(
        core,
        movements=movements,
        discharges=discharges,
        arrow_color="navy",
        arrow_width=2.0,
        discharge_color="red",
        title="Shuffling Scheme -- coreMaker Core",
    )
    fig4.savefig("coremaker_shuffle_map.png", dpi=150, bbox_inches="tight")
    print("Saved: coremaker_shuffle_map.png")


if __name__ == "__main__":
    main()
