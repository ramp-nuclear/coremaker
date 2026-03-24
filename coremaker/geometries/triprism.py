from typing import Self, Any, Type

import numpy as np

from ramp_core.serializable import Serializable, deserialize_default
from scipy.linalg import norm

from coremaker.surfaces import Plane
from coremaker.transform import Transform


def _form_prism(a, b, c):
    n1, n2, n3 = (np.cross(x.a, y.a) for x, y in [(a, b), (b, c), (a, c)])
    crosses = (np.cross(x, y) for x, y in [(n1, n2), (n2, n3), (n1, n3)])
    return all(np.allclose(v, 0, atol=1e-10) for v in crosses)


class TriPrism(Serializable):
    """Represent a triangular prism"""

    ser_identifier = "TriPrism"
    __slots__ = ("sides", "height", "center")

    def __init__(
        self,
        sides: tuple[Plane, Plane, Plane],
        height: float,
        center: tuple[float, float, float],
        check: bool = False,
    ):
        """

        Parameters
        ----------
        sides: tuple[Plane, Plane, Plane]
            The sides of the prism, which are three planes which intersect to create 3 parallel lines
        height: cm
            Height along the prismatic axis, which is the axis of the parallel intersection lines

        """
        if check:
            assert _form_prism(*sides)
        self.sides = tuple(sides)
        self.height = height
        self.center = tuple(center)

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {
            "sides": [side.serialize() for side in self.sides],
            "height": self.height,
            "center": list(self.center),
        }

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        sides = tuple(deserialize_default(t, supported=supported) for t in d["sides"])
        return cls(sides, d["height"], tuple(d["center"]))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.sides == other.sides and self.height == other.height and self.center == other.center

    def __hash__(self):
        return hash((self.sides, self.height, self.center))

    def __repr__(self):
        return (
            f"TriPrism<sides:{self.sides}, height:{self.height:.4g}, "
            f"center:({', '.join(f'{v:.4g}' for v in self.center)})"
        )

    @property
    def surfaces(self) -> tuple[Plane, Plane, Plane, Plane, Plane]:
        p1, p2 = self.sides[:2]
        perp = np.cross(np.array(p1.a), np.array(p2.a))
        perp /= norm(perp, ord=2)
        c = np.array(self.center)
        clen = norm(c, ord=2)
        costheta = np.dot(c / clen, perp)
        d1 = self.height + clen * costheta
        d2 = self.height - clen * costheta
        bottom = Plane(*perp, d1)
        top = -Plane(*perp, d2)
        bottom = bottom if bottom.check_point(self.center) == 1 else -bottom
        top = top if top.check_point(self.center) == -1 else -top

        return (*self.sides, bottom, top)

    @property
    def volume(self) -> float:
        dists = [side.distance(self.center) for side in self.sides]
        return self.height * _area(*dists)

    def transform(self: Self, transform: Transform) -> Self:
        """Get a transformed prism instead"""
        return type(self)(
            tuple(side.transform(transform) for side in self.sides),
            self.height,
            transform @ self.center,
        )


def _area(h1, h2, h3) -> float:
    """According to the area theorem on
    `Wikipedia <https://en.wikipedia.org/wiki/Altitude_(triangle)>`__.
    """
    invh = (1 / h1, 1 / h2, 1 / h3)
    h = sum(invh) / 2
    invarea = 4 * np.sqrt(np.product([h] + [(h - v) for v in invh]))
    return 1 / invarea
