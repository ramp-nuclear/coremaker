"""Burnup / depletion heatmap."""

from typing import Optional

from coremaker.visualization._engine import plot_heatmap

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_burnup_map(
    core,
    dep_map: Optional[dict[str, float]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlGnBu",
    units: str = "Fraction Depleted",
    title: str = "Burnup Map",
) -> tuple[Figure, plt.Axes]:
    """Plot a burnup / depletion heatmap.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    dep_map : dict, optional
        Mapping of site to depletion fraction, as returned by
        ``ramp.state_analysis.depletion.depletion_map``.
    ax : matplotlib Axes, optional
    cmap : str
        Colormap name (default: 'YlGnBu').
    units : str
        Colorbar label.
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]

    """
    return plot_heatmap(core, site_values=dep_map, cmap=cmap, units=units, title=title, ax=ax)
