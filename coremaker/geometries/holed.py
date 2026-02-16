"""A concrete implementation of the holed geometry protocol

"""
from typing import Sequence, Any, Type, TypeVar
try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

from ramp_core.serializable import Serializable, deserialize_default

from coremaker.protocols.geometry import Geometry
from coremaker.protocols.surface import Surface
from coremaker.transform import Transform


class ConcreteHoledGeometry(Serializable):
    """A geometry that is defined with some surfaces excluded.
    """

    ser_identifier = "HoledGeometry"
    __slots__ = ['inclusive', 'internal_exclusives', 'external_exclusives', 'exclusives']

    def __init__(self, inclusive: Geometry,
                 internal_exclusives: Sequence[Geometry],
                 external_exclusives: Sequence[Geometry] = ()):
        """
        Parameters
        ----------
        inclusive: Geometry
            The geometry things lie totally within
        internal_exclusives: Sequence[Geometry]
            The holes in the cheese.
        external_exclusives: Sequence[Geometry]
            The holes in the cheese that are sticking out of the cheese.
        """
        self.inclusive = inclusive
        self.internal_exclusives = internal_exclusives
        self.external_exclusives = external_exclusives
        self.exclusives = list(self.internal_exclusives) + list(self.external_exclusives)

    @property
    def volume(self) -> float | None:
        """The volume within the geometry.

        """
        if self.external_exclusives:
            return None
        try:
            return self.inclusive.volume - sum(g.volume for g in self.exclusives)
        except TypeError:
            return None

    @property
    def surfaces(self) -> list[Surface]:
        """Return a list of all surfaces, exclusive or inclusive.

        """
        return (list(self.inclusive.surfaces)
                + sum((list(g.surfaces) for g in self.exclusives), []))

    def __repr__(self) -> str:
        return f"HoledGeometry({self.inclusive}, {self.exclusives})"

    def __sub__(self, other: Geometry) -> "ConcreteHoledGeometry":
        return type(self)(self.inclusive, list(self.internal_exclusives) + [other], self.external_exclusives)

    def transform(self, transform: Transform) -> "ConcreteHoledGeometry":
        """Transform the geometry and all the holes inside.

        Parameters
        ----------
        transform: Transform
            The transformation to apply.

        """
        return type(self)(self.inclusive.transform(transform),
                          [g.transform(transform) for g in self.internal_exclusives],
                          [g.transform(transform) for g in self.external_exclusives])

    def _comparable(self):
        return {"inclusive": self.inclusive,
                "internal_exclusives": frozenset(self.internal_exclusives),
                "external_exclusives": frozenset(self.external_exclusives),
                }

    def serialize(self) -> tuple[str, dict[str, Any]]:
        data = dict(inclusive=self.inclusive.serialize(),
                    internal_exclusives=[g.serialize() for g in self.internal_exclusives],
                    external_exclusives=[g.serialize() for g in self.external_exclusives],
                    )
        return self.ser_identifier, data

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        inclusive = deserialize_default(d["inclusive"], supported=supported)
        internal = [deserialize_default(v, supported=supported) for v in d["internal_exclusives"]]
        external = [deserialize_default(v, supported=supported) for v in d["external_exclusives"]]
        return cls(inclusive=inclusive, internal_exclusives=internal, external_exclusives=external)

    def __eq__(self, other: Self):
        if isinstance(other, ConcreteHoledGeometry):
            return self._comparable() == other._comparable()
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self._comparable().values()))

