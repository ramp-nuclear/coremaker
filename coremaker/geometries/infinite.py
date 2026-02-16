"""The infinite geometry that knows no bounds.

"""

from dataclasses import dataclass

from ramp_core.serializable import Serializable

from coremaker.transform import Transform


@dataclass(frozen=True, init=False)
class _InfiniteGeometry(Serializable):
    """An infinite, surfaceless geometry

    """

    ser_identifier = "InfiniteGeometry"
    __slots__ = ()

    def __init__(self):
        pass

    @property
    def volume(self) -> None: return None

    def transform(self, transform: Transform) -> "_InfiniteGeometry":
        return self

    @property
    def surfaces(self) -> ():
        return ()

    def serialize(self) -> tuple[str, dict]: return self.ser_identifier, {}

    @classmethod
    def deserialize(cls, *_, **__) -> "_InfiniteGeometry": return infiniteGeometry

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return True
        return NotImplemented

    def __hash__(self): return 42  # Singleton object, we can cheat


infiniteGeometry = _InfiniteGeometry()


__all__ = ['infiniteGeometry']

