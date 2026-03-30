"""Built-in physics map functions for common reactor analysis scenarios."""

from coremaker.visualization.maps.burnup import plot_burnup_map
from coremaker.visualization.maps.component_type import plot_component_type_map
from coremaker.visualization.maps.power import plot_power_map
from coremaker.visualization.maps.shuffle import plot_shuffle_map

__all__ = [
    "plot_power_map",
    "plot_burnup_map",
    "plot_component_type_map",
    "plot_shuffle_map",
]
