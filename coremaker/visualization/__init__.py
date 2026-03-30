"""Core Visualization Library (CVL).

A modular library for automated generation of 2D radial core maps at the
homogenized assembly level.  All plotting functions accept a
``coremaker.core.Core`` object directly.

Install the visualization extra to use this module::

    pip install ramp-coremaker[viz]

"""

__all__ = [
    "plot_heatmap",
    "plot_categorical",
    "plot_arrows",
    "plot_power_map",
    "plot_burnup_map",
    "plot_component_type_map",
    "plot_shuffle_map",
]


def __getattr__(name):
    """Lazy-load plotting functions to avoid importing matplotlib at package import time."""
    _engine_funcs = {"plot_heatmap", "plot_categorical", "plot_arrows"}
    _map_funcs = {
        "plot_power_map": "coremaker.visualization.maps.power",
        "plot_burnup_map": "coremaker.visualization.maps.burnup",
        "plot_component_type_map": "coremaker.visualization.maps.component_type",
        "plot_shuffle_map": "coremaker.visualization.maps.shuffle",
    }

    if name in _engine_funcs:
        from coremaker.visualization import _engine

        return getattr(_engine, name)
    if name in _map_funcs:
        import importlib

        mod = importlib.import_module(_map_funcs[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
