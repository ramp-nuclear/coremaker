"""Transition map showing rod movements, loads, and discharges."""

from dataclasses import dataclass, field
from typing import Optional

from coremaker.visualization.coregeometry import all_site_geometries

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch
except ImportError as e:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install ramp-coremaker[viz]"
    ) from e


@dataclass
class TransitionPlan:
    """Structured description of all transitions in a shuffling step.

    Parameters
    ----------
    movements : list[tuple[str, str]]
        Pairs of (source, destination) site labels for rod movements.
    loads : list[tuple[str, str]]
        Pairs of (site, rod_name) for freshly loaded rods.
    discharges : list[str]
        Site labels from which rods are removed.

    """

    movements: list[tuple[str, str]] = field(default_factory=list)
    loads: list[tuple[str, str | None]] = field(default_factory=list)
    discharges: list[str] = field(default_factory=list)


def plan_from_scheme(scheme) -> TransitionPlan:
    """Build a TransitionPlan from a coreoperator Scheme.

    Parameters
    ----------
    scheme : coreoperator.mobilization.Scheme

    Returns
    -------
    TransitionPlan

    """
    from coreoperator.mobilization import CyclicShuffle, LoadChain, LoadSite, Remove

    plan = TransitionPlan()
    for action in scheme.actions:
        if isinstance(action, CyclicShuffle):
            sites = [site for site, _transform in action.sites]
            for i in range(len(sites)):
                plan.movements.append((sites[i], sites[(i + 1) % len(sites)]))

        elif isinstance(action, LoadSite):
            site, _transform = action.sites[0]
            rod = action.factory()
            plan.loads.append((site, getattr(rod, "type_key", None)))

        elif isinstance(action, LoadChain):
            sites = [site for site, _transform in action.sites]
            rod = action.factory()
            plan.loads.append((sites[0], getattr(rod, "type_key", None)))
            for i in range(len(sites) - 1):
                plan.movements.append((sites[i], sites[i + 1]))
            plan.discharges.append(sites[-1])

        elif isinstance(action, Remove):
            for site in action.sites:
                plan.discharges.append(site)

    return plan


def plot_transition(
    core,
    plan: TransitionPlan,
    *,
    ax: Optional[plt.Axes] = None,
    arrow_color: str = "black",
    arrow_width: float = 1.5,
    load_marker: str = "s",
    load_markersize: float = 14,
    load_color_dict: Optional[dict[str, str]] = None,
    rod_color_dict: Optional[dict[str, str]] = None,
    discharge_color: str = "red",
    discharge_marker: str = "X",
    discharge_markersize: float = 16,
    title: str = "Shuffling Scheme",
    **rod_map_kwargs,
) -> tuple[Figure, plt.Axes]:
    """Render a transition map showing movements, loads, and discharges.

    The base layer is a rod-type map (``plot_rod_map``) so each site is
    coloured by its rod's ``type_key``.  Load markers are coloured to
    match their rod type and a legend on the left identifies them.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    plan : TransitionPlan
        Structured transition data (movements, loads, discharges).
    ax : matplotlib Axes, optional
        If None, a new figure with a rod map base is created.
    arrow_color : str
        Color for movement arrows.
    arrow_width : float
        Arrow line width.
    load_marker : str
        Marker style for loaded sites (default: 's' square).
    load_markersize : float
        Size of load markers.
    load_color_dict : dict[str, str], optional
        Mapping of rod type name to color for load markers.
        Auto-assigned from ``gist_rainbow`` if not provided.
    rod_color_dict : dict[str, str], optional
        Mapping of rod type name to color for the base rod map.
        Auto-assigned from ``tab20`` if not provided.
    discharge_color : str
        Color for discharge markers.
    discharge_marker : str
        Marker style for discharged sites (default: 'X').
    discharge_markersize : float
        Size of discharge markers.
    title : str
        Plot title.
    **rod_map_kwargs
        Forwarded to ``plot_rod_map`` (e.g. *site_names*).

    Returns
    -------
    tuple[Figure, Axes]

    """
    from coremaker.visualization.maps.categorical import plot_rod_map

    geometries = all_site_geometries(core)

    if ax is None:
        fig, ax = plot_rod_map(core, title=title, color_dict=rod_color_dict, **rod_map_kwargs)
    else:
        fig = ax.get_figure()

    # Draw movement arrows with slight curvature to distinguish overlapping paths
    for i, (src, dst) in enumerate(plan.movements):
        if src in geometries and dst in geometries:
            src_geom = geometries[src]
            dst_geom = geometries[dst]
            rad = 0.15
            arrow = FancyArrowPatch(
                (src_geom.center_x, src_geom.center_y),
                (dst_geom.center_x, dst_geom.center_y),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=15,
                color=arrow_color,
                linewidth=arrow_width,
                zorder=10,
            )
            ax.add_patch(arrow)

    # Build color mapping for loaded rod types
    load_types = sorted({name for _, name in plan.loads if name is not None})
    if load_color_dict is None:
        tab10 = plt.get_cmap("gist_rainbow")
        load_color_dict = {name: tab10(i / max(len(load_types), 1)) for i, name in enumerate(load_types)}

    # Draw load markers coloured by rod type
    for site, rod_name in plan.loads:
        if site in geometries:
            geom = geometries[site]
            color = load_color_dict.get(rod_name, "gray") if rod_name is not None else "gray"
            ax.plot(
                geom.center_x,
                geom.center_y,
                marker=load_marker,
                color=color,
                markersize=load_markersize,
                markeredgewidth=2.5,
                fillstyle="none",
                zorder=11,
            )

    # Legend for loaded rod types (left side)
    # Preserve the existing rod-map legend before adding the load legend
    from matplotlib.lines import Line2D

    existing_legend = ax.get_legend()
    if load_types:
        legend_handles = [
            Line2D(
                [0], [0],
                marker=load_marker,
                color=load_color_dict[name],
                markersize=10,
                markeredgewidth=2.5,
                fillstyle="none",
                linestyle="None",
                label=name,
            )
            for name in load_types
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    if existing_legend is not None:
        ax.add_artist(existing_legend)

    # Draw discharge markers
    for site in plan.discharges:
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


def plot_scheme(
    core,
    scheme,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Shuffling Scheme",
    **kwargs,
) -> tuple[Figure, plt.Axes]:
    """Render a transition map directly from a coreoperator Scheme.

    Convenience wrapper that calls ``plan_from_scheme`` then ``plot_transition``.

    Parameters
    ----------
    core : coremaker.core.Core
        A coreMaker Core object.
    scheme : coreoperator.mobilization.Scheme
        A shuffling scheme.
    ax : matplotlib Axes, optional
    title : str
        Plot title.
    **kwargs
        Forwarded to ``plot_transition``.

    Returns
    -------
    tuple[Figure, Axes]

    """
    plan = plan_from_scheme(scheme)
    return plot_transition(core, plan, ax=ax, title=title, **kwargs)
