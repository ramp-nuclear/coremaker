"""Defines the concept of a bare geometry for the necessities of general core object
representations

"""
from dataclasses import dataclass
from itertools import chain
from typing import Any, ClassVar, Sequence, Type, TypeVar

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

from ramp_core.serializable import Serializable, deserialize_default

from coremaker.geometries.holed import ConcreteHoledGeometry
from coremaker.protocols.geometry import Geometry
from coremaker.protocols.surface import Surface
from coremaker.surfaces.plane import Plane, calculate_plane_intersection_volume
from coremaker.transform import Transform


@dataclass(frozen=True)
class BareGeometry:
    """A geometry where the only known things are the external surfaces."""

    _surfaces: Sequence[Surface]
    _volume: float | None = None

    ser_identifier: ClassVar[str] = "BareGeometry"

    @property
    def volume(self) -> float | None:
        """Unless directly specified, the volume of a bare geometry is
        undefined."""
        if self._volume is not None:
            return self._volume
        if all(isinstance(s, Plane) for s in self.surfaces):
            # Safe because we check those are planes first
            return calculate_plane_intersection_volume(self.surfaces)  # type: ignore

    @property
    def surfaces(self) -> Sequence[Surface]:
        """A property to get the underlying surfaces."""
        return self._surfaces

    def transform(self, transform: Transform) -> "BareGeometry":
        """Return a geometry where all the surfaces have the same transform
        applied to them.

        Parameters
        ----------
        transform: Transform
            Transform to apply on the surfaces

        """
        return BareGeometry(
            [s.transform(transform) for s in self._surfaces], _volume=self._volume
        )

    def __and__(self, other: Geometry) -> "BareGeometry":
        return BareGeometry(list(chain(self._surfaces, other.surfaces)))

    def __sub__(self, other: Geometry) -> "ConcreteHoledGeometry":
        return ConcreteHoledGeometry(self, [other])

    def __eq__(self, other) -> bool:
        if isinstance(other, Geometry):
            return frozenset(self.surfaces) == frozenset(other.surfaces)
        return NotImplemented

    def __hash__(self):
        return hash((tuple(self.surfaces), self.volume))

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {"surfaces": [s.serialize() for s in self._surfaces],
                                     "volume": self._volume}

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        # We assume the data given is actually made up of surfaces, so we can ignore the typing for surfaces
        return cls(tuple(deserialize_default(t, supported=supported) for t in d["surfaces"]),  # type: ignore
                   d["volume"])

    def __repr__(self) -> str:
        return f"BareGeometry<{tuple(self.surfaces)}>"
