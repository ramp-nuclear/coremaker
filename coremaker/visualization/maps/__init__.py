"""Built-in map functions for common reactor analysis scenarios."""

__all__ = [
    "plot_heatmap",
    "plot_power_map",
    "plot_categorical",
    "plot_rod_map",
    "plot_transition",
    "plot_scheme",
    "TransitionPlan",
    "plan_from_scheme",
]


def __getattr__(name):
    """Lazy-load map functions to avoid importing coreoperator at package import time."""
    _map_funcs = {
        "plot_heatmap": "coremaker.visualization.maps.heatmap",
        "plot_power_map": "coremaker.visualization.maps.heatmap",
        "plot_categorical": "coremaker.visualization.maps.categorical",
        "plot_rod_map": "coremaker.visualization.maps.categorical",
        "plot_transition": "coremaker.visualization.maps.transition",
        "plot_scheme": "coremaker.visualization.maps.transition",
        "TransitionPlan": "coremaker.visualization.maps.transition",
        "plan_from_scheme": "coremaker.visualization.maps.transition",
    }

    if name in _map_funcs:
        import importlib

        mod = importlib.import_module(_map_funcs[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
