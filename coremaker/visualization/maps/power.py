"""Power distribution map with warm colors and optional peak annotation."""

from collections import defaultdict
from typing import Optional

from coremaker.visualization._core_geometry import all_site_geometries
from coremaker.visualization._engine import plot_heatmap

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


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
