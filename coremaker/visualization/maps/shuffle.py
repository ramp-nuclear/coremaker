"""Shuffling scheme map with movement arrows and discharge markers."""

from typing import Optional

from coremaker.visualization._engine import plot_arrows

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


def plot_shuffle_map(
    core,
    movements: Optional[list[tuple[str, str]]] = None,
    discharges: Optional[list[str]] = None,
    scheme=None,
    *,
    ax: Optional[plt.Axes] = None,
    arrow_color: str = "black",
    arrow_width: float = 1.5,
    discharge_color: str = "red",
    discharge_marker: str = "X",
    discharge_markersize: float = 16,
    title: str = "Shuffling Scheme",
) -> tuple[Figure, plt.Axes]:
    r"""Plot movement vectors and discharge markers for a shuffling scheme.

    Depicts arrows :math:`\vec{r}_{initial} \to \vec{r}_{final}` of fuel
    assemblies between operational cycles. Discharged rods (removed from core)
    are marked with an X (or custom marker).

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    movements : list of (source, dest) pairs, optional
        If not given and *scheme* is provided, extracted from the scheme.
    discharges : list of site labels, optional
        Sites from which rods are discharged. If not given and *scheme* is
        provided, extracted from the scheme.
    scheme : coreoperator.mobilization.Scheme, optional
        If provided, LoadChain actions produce movement arrows and Remove
        actions produce discharge markers.
    ax : matplotlib Axes, optional
    arrow_color : str
        Color for the movement arrows.
    arrow_width : float
        Arrow line width.
    discharge_color : str
        Color for discharge markers (default: red).
    discharge_marker : str
        Marker style for discharged sites (default: 'X').
    discharge_markersize : float
        Size of discharge markers.
    title : str
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]

    """
    if scheme is not None and movements is None and discharges is None:
        from coreoperator.mobilization import LoadChain, Remove

        movements = []
        discharges = []
        for action in scheme.actions:
            if isinstance(action, LoadChain):
                sites = [site for site, _transform in action.sites]
                for i in range(1, len(sites)):
                    movements.append((sites[i], sites[i - 1]))
            elif isinstance(action, Remove):
                for site in action.sites:
                    discharges.append(site)

    return plot_arrows(
        core,
        movements=movements,
        discharges=discharges,
        ax=ax,
        arrow_color=arrow_color,
        arrow_width=arrow_width,
        discharge_color=discharge_color,
        discharge_marker=discharge_marker,
        discharge_markersize=discharge_markersize,
        title=title,
    )
