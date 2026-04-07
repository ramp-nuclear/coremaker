"""Continuous heatmap rendering for core maps (power, burnup, etc.)."""

from collections import defaultdict
from typing import Optional

import numpy as np

from coremaker.visualization.coregeometry import (
    all_site_geometries,
    occupied_sites,
    site_labels_from_core,
)
from coremaker.visualization.geometry import make_patch

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_heatmap(
    core,
    site_values: dict[str, float],
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    units: str = "",
    title: str = "",
    show_labels: bool = True,
    label_fontsize: int = 7,
    empty_color: str = "#dddddd",
) -> tuple[Figure, plt.Axes]:
    """Render a continuous heatmap over a core geometry.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    site_values : dict[str, float]
        Scalar field values per site (power, burnup, etc.).
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colorbar limits. Auto-computed if not given.
    units : str
        Label for the colorbar.
    title : str
        Axes title.
    show_labels : bool
        Whether to annotate each cell with its site name.
    label_fontsize : int
        Font size for site labels.
    empty_color : str
        Color for unoccupied sites.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    geometries = all_site_geometries(core)
    occupied = occupied_sites(core)
    labels = site_labels_from_core(core)
    sites = list(core.grid.sites())

    # Resolve values
    values = {site: site_values[site] for site in sites if site in occupied and site in site_values}

    # Build patches and collect colors
    patches = []
    color_values = []
    for site in sites:
        geom = geometries[site]
        patches.append(make_patch(geom))
        color_values.append(values.get(site, np.nan))

    all_vals = [c for c in color_values if not np.isnan(c)]
    if vmin is None and all_vals:
        vmin = min(all_vals)
    if vmax is None and all_vals:
        vmax = max(all_vals)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    face_colors = []
    for c in color_values:
        if np.isnan(c):
            face_colors.append(empty_color)
        else:
            face_colors.append(colormap(norm(c)))

    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(face_colors)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)
    ax.add_collection(pc)

    # Colorbar
    if all_vals:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(units)

    # Site labels with numeric values
    if show_labels:
        for site in sites:
            geom = geometries[site]
            label = labels.get(site, site)
            val = values.get(site)
            if val is not None and not np.isnan(val):
                label = f"{label}\n{val:.2f}"
            ax.annotate(
                label,
                (geom.center_x, geom.center_y),
                ha="center",
                va="center",
                fontsize=label_fontsize,
            )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    return fig, ax


def plot_power_map(
    core,
    energy_map: Optional[dict[str, tuple[float, float]]] = None,
    site_values: Optional[dict[str, float]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    highlight_peaking: bool = True,
    cmap: str = "YlOrRd",
    units: str = "Relative Power",
    title: str = "Thermal Power Map",
) -> tuple[Figure, plt.Axes]:
    """Plot a power density / RPF heatmap with warm colors.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    energy_map : dict, optional
        Mapping of component path to (power_value, error).  If provided,
        aggregates per-component power to per-site values using
        ``split_name()``.
    site_values : dict[str, float], optional
        Pre-computed per-site power values.  Ignored if *energy_map* is given.
    ax : matplotlib Axes, optional
    highlight_peaking : bool
        If True, marks the peak-power assembly with a star marker.
    cmap : str
        Colormap name (default: 'YlOrRd' for warm colors).
    units : str
        Colorbar label.
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if energy_map is None and site_values is None:
        raise ValueError("Either energy_map or site_values must be provided.")

    if energy_map is not None:
        from ramp.state_analysis.util import split_name

        site_powers: dict[str, list[float]] = defaultdict(list)
        for comp_path, (value, _error) in energy_map.items():
            site, *_ = split_name(comp_path)
            if site is not None:
                site_powers[site].append(value)
        site_values = {site: sum(vals) / len(vals) for site, vals in site_powers.items()}

    fig, ax = plot_heatmap(core, site_values=site_values, cmap=cmap, units=units, title=title, ax=ax)

    if highlight_peaking and site_values:
        peak_site = max(site_values, key=site_values.get)
        geometries = all_site_geometries(core)
        geom = geometries[peak_site]
        ax.plot(
            geom.center_x,
            geom.center_y,
            marker="*",
            color="black",
            markersize=12,
            zorder=5,
        )

    return fig, ax
