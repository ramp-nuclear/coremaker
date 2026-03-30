"""Categorical component type map with discrete colors."""

from typing import Callable, Optional

from coremaker.visualization._core_geometry import occupied_sites
from coremaker.visualization._engine import plot_categorical

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_component_type_map(
    core,
    type_fn: Optional[Callable] = None,
    site_categories: Optional[dict[str, str]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    color_dict: Optional[dict[str, str]] = None,
    title: str = "Component Types",
) -> tuple[Figure, plt.Axes]:
    """Plot a categorical map showing assembly / material types.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    type_fn : callable, optional
        Function with signature ``type_fn(site: str, element) -> str``
        returning a category label for each occupied site.
    site_categories : dict[str, str], optional
        Pre-computed category labels per site.  Ignored if *type_fn* is given.
    ax : matplotlib Axes, optional
    color_dict : dict, optional
        Category name to color mapping.
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if type_fn is not None:
        grid = core.grid
        occupied = occupied_sites(core)
        site_categories = {site: type_fn(site, grid[site]) for site in occupied}

    return plot_categorical(
        core,
        site_categories=site_categories,
        ax=ax,
        color_dict=color_dict,
        title=title,
    )
