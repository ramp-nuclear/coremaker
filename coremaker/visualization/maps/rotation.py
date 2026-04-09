"""Rotation map showing the spatial orientation of each rod."""

from typing import Optional

import numpy as np

from coremaker.visualization.coregeometry import (
    all_site_geometries,
    occupied_sites,
    site_labels_from_core,
    site_rotations,
)
from coremaker.visualization.geometry import make_patch

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.figure import Figure
except ImportError as e:
    e.add_note("matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]")
    raise


def _draw_rotation_arrows(
    ax: plt.Axes,
    geometries: dict,
    rotations: dict[str, float],
    *,
    arrow_color: str = "black",
    arrow_width: float = 1.5,
    arrow_length_frac: float = 0.35,
    zorder: int = 5,
) -> None:
    """Draw rotation-indicating arrows on *ax* for each site in *rotations*.

    Parameters
    ----------
    ax : matplotlib Axes
    geometries : dict[str, CellGeometry]
    rotations : dict[str, float]
        Mapping of site label to rotation angle in degrees (0 = up/north).
    arrow_color : str
    arrow_width : float
    arrow_length_frac : float
        Arrow length as a fraction of half the cell's smaller dimension.
    zorder : int

    """
    for site, angle_deg in rotations.items():
        if site not in geometries:
            continue
        geom = geometries[site]
        half_size = min(geom.width_x, geom.width_y or geom.width_x) / 2
        length = arrow_length_frac * half_size

        angle_rad = np.radians(angle_deg)
        dx = -length * np.sin(angle_rad)
        dy = length * np.cos(angle_rad)

        cell_h = geom.width_y or geom.width_x
        cx = geom.center_x
        cy = geom.center_y - cell_h * 0.2
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=arrow_width,
            ),
            zorder=zorder,
        )


def plot_rotation_map(
    core,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Rotation Map",
    arrow_length_frac: float = 0.35,
    arrow_color: str = "black",
    arrow_width: float = 1.5,
    show_labels: bool = True,
    label_fontsize: int = 7,
    empty_color: str = "#dddddd",
    occupied_color: str = "#bbbbbb",
    show_angle_text: bool = False,
) -> tuple[Figure, plt.Axes]:
    """Plot the spatial orientation of each rod as directional arrows.

    Each occupied site is drawn as a uniform cell with an arrow indicating
    the rod's z-rotation (0 deg = north/up, positive = counter-clockwise).

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    title : str
        Plot title.
    arrow_length_frac : float
        Arrow length as a fraction of half the cell's smaller dimension.
    arrow_color : str
        Color of rotation arrows.
    arrow_width : float
        Line width of rotation arrows.
    show_labels : bool
        Whether to annotate each cell with its site name.
    label_fontsize : int
        Font size for site labels.
    empty_color : str
        Color for unoccupied sites.
    occupied_color : str
        Color for occupied sites.
    show_angle_text : bool
        Whether to display the angle value as text below each arrow.

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
    rotations = site_rotations(core)
    sites = list(core.grid.sites())

    patches = []
    face_colors = []
    for site in sites:
        patches.append(make_patch(geometries[site]))
        face_colors.append(occupied_color if site in occupied else empty_color)

    pc = PatchCollection(patches, match_original=False)
    pc.set_facecolor(face_colors)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)
    ax.add_collection(pc)

    _draw_rotation_arrows(
        ax,
        geometries,
        rotations,
        arrow_color=arrow_color,
        arrow_width=arrow_width,
        arrow_length_frac=arrow_length_frac,
    )

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

    if show_angle_text:
        for site in sites:
            if site in rotations:
                geom = geometries[site]
                cell_h = geom.width_y or geom.width_x
                ax.annotate(
                    f"{rotations[site]:.0f}\u00b0",
                    (geom.center_x, geom.center_y - cell_h * 0.35),
                    ha="center",
                    va="center",
                    fontsize=label_fontsize - 1,
                    color="gray",
                )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    return fig, ax
