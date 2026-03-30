"""Categorical (discrete) color map rendering for core maps."""

from typing import Optional

from coremaker.visualization.coregeometry import (
    all_site_geometries,
    occupied_sites,
    site_labels_from_core,
)
from coremaker.visualization.geometry import make_patch

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_categorical(
    core,
    site_categories: dict[str, str],
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
    site_categories : dict[str, str]
        Categorical label per occupied site (e.g. assembly type).
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
    labels = site_labels_from_core(core)
    sites = list(core.grid.sites())

    # Build color dict if not provided
    unique_cats = sorted(set(site_categories.values()))
    if color_dict is None:
        tab20 = plt.get_cmap("tab20")
        color_dict = {cat: tab20(i / max(len(unique_cats), 1)) for i, cat in enumerate(unique_cats)}

    # Build patches
    patches = []
    face_colors = []
    for site in sites:
        geom = geometries[site]
        patches.append(make_patch(geom))
        cat = site_categories.get(site)
        face_colors.append(color_dict.get(cat, empty_color) if cat else empty_color)

    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(face_colors)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)
    ax.add_collection(pc)

    # Legend
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


def plot_rod_map(
    core,
    site_names: Optional[dict[str, str]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    color_dict: Optional[dict[str, str]] = None,
    title: str = "Rod Map",
    show_labels: bool = True,
    label_fontsize: int = 7,
    empty_color: str = "#dddddd",
) -> tuple[Figure, plt.Axes]:
    """Plot a categorical rod map coloured by rod type.

    By default the category for each site is taken from the element's
    ``type_key`` attribute.  Sites whose element has ``type_key is None``
    are left unnamed (rendered with *empty_color*).  An optional
    *site_names* mapping can override or supplement the automatic names.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    site_names : dict[str, str], optional
        Explicit mapping of site label to rod name.  Entries here take
        precedence over ``type_key``.
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    color_dict : dict[str, str], optional
        Mapping of rod name to color.  Auto-assigned if not provided.
    title : str
        Plot title.
    show_labels : bool
        Whether to annotate each cell with its site name.
    label_fontsize : int
        Font size for site labels.
    empty_color : str
        Color for unoccupied or unnamed sites.

    Returns
    -------
    tuple[Figure, Axes]

    """
    grid = core.grid
    occupied = occupied_sites(core)
    if site_names is None:
        site_names = {}

    site_categories = {}
    for site in occupied:
        if site in site_names:
            site_categories[site] = site_names[site]
        else:
            element = grid[site]
            type_key = getattr(element, "type_key", None)
            if type_key is not None:
                site_categories[site] = type_key

    return plot_categorical(
        core,
        site_categories,
        ax=ax,
        color_dict=color_dict,
        title=title,
        show_labels=show_labels,
        label_fontsize=label_fontsize,
        empty_color=empty_color,
    )
