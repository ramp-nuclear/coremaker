"""Generic heatmap and categorical rendering engine for core maps."""

from typing import Callable, Optional

import numpy as np

from coremaker.visualization._core_geometry import (
    all_site_geometries,
    occupied_sites,
    site_labels_from_core,
)
from coremaker.visualization._geometry import make_patch

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_heatmap(
    core,
    site_values: Optional[dict[str, float]] = None,
    value_fn: Optional[Callable[[str], float]] = None,
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
    site_values : dict[str, float], optional
        Scalar field values per site (power, burnup, etc.).
    value_fn : callable, optional
        If provided, called as ``value_fn(site) -> float`` for each occupied
        site.  Overrides *site_values*.
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
    if site_values is None:
        site_values = {}

    # Resolve values
    values = {}
    for site in sites:
        if site in occupied:
            if value_fn is not None:
                values[site] = value_fn(site)
            elif site in site_values:
                values[site] = site_values[site]

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


def plot_categorical(
    core,
    site_categories: Optional[dict[str, str]] = None,
    category_fn: Optional[Callable[[str], str]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    color_dict: Optional[dict[str, str]] = None,
    title: str = "",
    show_labels: bool = True,
    label_fontsize: int = 7,
    empty_color: str = "#dddddd",
) -> tuple[Figure, plt.Axes]:
    """Render a categorical (discrete) color map over a core geometry.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    site_categories : dict[str, str], optional
        Categorical field per site (assembly types).
    category_fn : callable, optional
        If provided, called as ``category_fn(site) -> str`` for each occupied
        site.  Overrides *site_categories*.
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    color_dict : dict[str, str], optional
        Mapping of category name to color. If not provided, colors are
        auto-assigned from the ``tab20`` colormap.
    title : str
        Axes title.
    show_labels : bool
        Whether to annotate each cell with its site name.
    label_fontsize : int
        Font size for site labels.
    empty_color : str
        Color for unoccupied or uncategorized sites.

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
    if site_categories is None:
        site_categories = {}

    # Resolve categories
    categories = {}
    for site in sites:
        if site in occupied:
            if category_fn is not None:
                categories[site] = category_fn(site)
            elif site in site_categories:
                categories[site] = site_categories[site]

    # Build color dict if not provided
    unique_cats = sorted(set(categories.values()))
    if color_dict is None:
        tab20 = plt.get_cmap("tab20")
        color_dict = {cat: tab20(i / max(len(unique_cats), 1)) for i, cat in enumerate(unique_cats)}

    # Build patches
    patches = []
    face_colors = []
    for site in sites:
        geom = geometries[site]
        patches.append(make_patch(geom))
        cat = categories.get(site)
        face_colors.append(color_dict.get(cat, empty_color) if cat else empty_color)

    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(face_colors)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)
    ax.add_collection(pc)

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [Patch(facecolor=color_dict[cat], edgecolor="black", label=cat) for cat in unique_cats]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # Site labels
    if show_labels:
        for site in sites:
            geom = geometries[site]
            label = labels.get(site, site)
            ax.annotate(
                label,
                (geom.center_x, geom.center_y),
                ha="center",
                va="center",
                fontsize=label_fontsize,
            )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title or "Component Types")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    return fig, ax


def plot_arrows(
    core,
    movements: Optional[list[tuple[str, str]]] = None,
    discharges: Optional[list[str]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    arrow_color: str = "black",
    arrow_width: float = 1.5,
    discharge_color: str = "red",
    discharge_marker: str = "X",
    discharge_markersize: float = 16,
    base_cmap: str = "Pastel1_r",
    title: str = "Shuffling Scheme",
) -> tuple[Figure, plt.Axes]:
    """Render shuffle movement arrows and discharge markers over a core map.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    movements : list of (source, dest) pairs, optional
        Pairs of site labels for movement arrows.
    discharges : list of site labels, optional
        Sites from which rods are discharged (removed from core).
    ax : matplotlib Axes, optional
        If None, a new figure with a light base map is created.
    arrow_color : str
        Color for the movement arrows.
    arrow_width : float
        Arrow line width.
    discharge_color : str
        Color for the discharge markers.
    discharge_marker : str
        Marker style for discharged sites (default: 'X').
    discharge_markersize : float
        Size of discharge markers.
    base_cmap : str
        Colormap for the light base heatmap (used if no axes provided).
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if movements is None:
        movements = []
    if discharges is None:
        discharges = []

    geometries = all_site_geometries(core)
    occupied = occupied_sites(core)

    # Draw a light base map first if no axes given
    if ax is None:
        dummy_values = {site: 0.5 for site in occupied}
        fig, ax = plot_heatmap(
            core,
            site_values=dummy_values,
            cmap=base_cmap,
            show_labels=True,
            title=title,
        )
        # Remove the colorbar from the base map
        if ax.get_figure().axes and len(ax.get_figure().axes) > 1:
            ax.get_figure().axes[-1].remove()
    else:
        fig = ax.get_figure()

    # Draw arrows for movements
    for src, dst in movements:
        if src in geometries and dst in geometries:
            src_geom = geometries[src]
            dst_geom = geometries[dst]
            arrow = FancyArrowPatch(
                (src_geom.center_x, src_geom.center_y),
                (dst_geom.center_x, dst_geom.center_y),
                arrowstyle="-|>",
                mutation_scale=15,
                color=arrow_color,
                linewidth=arrow_width,
                zorder=10,
            )
            ax.add_patch(arrow)

    # Draw discharge markers
    for site in discharges:
        if site in geometries:
            geom = geometries[site]
            ax.plot(
                geom.center_x,
                geom.center_y,
                marker=discharge_marker,
                color=discharge_color,
                markersize=discharge_markersize,
                markeredgewidth=2.5,
                zorder=11,
            )

    return fig, ax
