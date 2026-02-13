"""The infinite geometry that knows no bounds.

"""

from dataclasses import dataclass

from coremaker.transform import Transform


@dataclass(frozen=True, init=False)
class _InfiniteGeometry:
    """An infinite, surfaceless geometry

    """

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


infiniteGeometry = _InfiniteGeometry()


__all__ = ['infiniteGeometry']
