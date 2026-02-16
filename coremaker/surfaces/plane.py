"""A planar surface.

"""
import math
from functools import lru_cache
from typing import Any, Type, TypeVar, Sequence

from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from ramp_core.serializable import Serializable
from scipy.linalg import norm

from coremaker.protocols.surface import Surface
from coremaker.transform import Transform
from coremaker.units import cm


class Plane(Serializable):
    r"""A half-plane with the equation:

    .. math:: a_1x+a_2y+a_3z \geq b

    For most method explanations, see the :class:`coremaker.protocols.surface.Surface` protocol.
    """

    ser_identifier = "Plane"
    __slots__ = ['a', 'b']

    @lru_cache(maxsize=100)
    def __new__(cls, *args, **kwargs):
        """Creates a new Plane object, or gives a cached one.

        The reason we cache this is that it was found this mattered a lot in certain applications.
        We haven't optimized the cache size yet.
        100 and 500 had similar run times and memory footprint.

        This is likely to not matter in the long run once we have a depleted core object.

        """
        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __getnewargs__(self):
        return *self.a, self.b

    def isclose(self, other: Surface) -> bool:
        """Checks whether two planes are close to one another.

        Planes are considered close if their coefficients in the equation are close.

        Parameters
        ----------
        other: Plane
            The other plane to compare to.

        """
        if not isinstance(other, Plane):
            return False
        coeff1, coeff2 = np.array(self.a), np.array(other.a)
        norm1, norm2 = norm(coeff1, ord=2), norm(coeff2, ord=2)
        b1, b2 = self.b / norm1, other.b / norm2
        coeff1, coeff2 = coeff1 / norm1, coeff2 / norm2
        return (math.isclose(b1, b2, rel_tol=0, abs_tol=1e-5)
                and math.isclose(coeff1 @ coeff2, 1, rel_tol=1e-10, abs_tol=0)
                )

    def __init__(self, /, a1: cm, a2: cm, a3: cm, b: cm, *, check=False):
        self.a = (a1, a2, a3)
        if check and not norm(self.a, ord=2):
            raise ValueError("Plane cannot have a_1=a_2=a_3=0")
        self.b = b

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return self.ser_identifier, {"coeffs": [*self.a, self.b]}

    @classmethod
    def deserialize(cls: Type[Self], d: dict[str, Any], *, supported: dict[str, Type[Serializable]]) -> Self:
        return cls(*d["coeffs"])

    def __hash__(self):
        return hash((self.a, self.b))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.a == other.a and self.b == other.b
        return NotImplemented

    def normal(self, point: tuple[cm, cm, cm]) -> tuple[float, float, float]:
        n = norm(self.a, ord=2)
        if not n:
            raise ValueError("Cannot have a normal to a surface with a1=a2=a3=0")
        # Safe because we know self.a has length 3
        return tuple(-np.array(self.a) / n)  # type: ignore

    normal.__doc__ = Surface.normal.__doc__

    def __neg__(self) -> "Plane":
        return Plane(*(-v for v in self.a), -self.b)

    def transform(self, transform: Transform) -> "Plane":
        a = np.array(self.a, dtype=float)
        a_new = transform.rotmat @ a
        b_new = self.b + (a_new @ transform.translation)
        return Plane(*a_new, b_new.item(0))

    transform.__doc__ = Surface.transform.__doc__

    def __repr__(self) -> str:
        a1, a2, a3 = self.a
        if not any(self.a):
            return "Plane<Undefined plane with a=0!>"
        if a1 != 0 and self.isclose(Plane(a1, 0.0, 0.0, self.b)):
            sign = '-' if a1 < 0 else ''
            return f"Plane<{sign}x >= {abs(self.b / a1):.3e}>"
        elif a2 != 0 and self.isclose(Plane(0.0, a2, 0.0, self.b)):
            sign = '-' if a2 < 0 else ''
            return f"Plane<{sign}y >= {abs(self.b / a2):.3e}>"
        elif a3 != 0 and self.isclose(Plane(0.0, 0.0, a3, self.b)):
            sign = '-' if a3 < 0 else ''
            return f"Plane<{sign}z >= {abs(self.b / a3):.3e}>"
        elif self.isclose(Plane(0.0, a2, a3, self.b)):
            return f"Plane<{a2:.3e}y + {a3:.3e}z " \
                   f">= {self.b:.3e}>"
        elif self.isclose(Plane(a1, 0.0, a3, self.b)):
            return f"Plane<{a1:.3e}x + {a3:.3e}z " \
                   f">= {self.b:.3e}>"
        elif self.isclose(Plane(a1, a2, 0.0, self.b)):
            return f"Plane<{a1:.3e}x + {a2:.3e}y " \
                   f">= {self.b:.3e}>"
        else:
            return f"Plane<{a1:.3e}x + {a2:.3e}y + {a3:.3e}z " \
                   f">= {self.b:.3e}>"


def calculate_plane_intersection_volume(planes: Sequence[Plane]) -> float | None:
    """Calculate volume of a solid composed of an intersection of planes (halfspaces).
    returns None when the dimension of the intersection is less than 3 or if it has infinite extents.
    Parameters
    ----------
    planes - the sequence of planes

    Returns
    -------
    float | None
        The volume of the resulting solid or None if the solid is either empty or has infinite extents
    """
    # polyhedra with finite volume have at least 4 faces
    if len(planes) < 4:
        return None
    halfspaces = -np.array([[*p.a, -p.b] for p in planes])
    # check that the halfspaces span the whole 3d space
    if np.linalg.matrix_rank(halfspaces[:, :-1]) < 3:
        return None
    norm_vector = np.linalg.norm(halfspaces[:, :-1], axis=1)[:, np.newaxis]
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    # noinspection PyDeprecation
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    # check that the intersection is not empty
    if not res.success:
        return None
    radius = res.x[-1]
    # check that the intersection is 3d
    if radius <= 0.0:
        return None
    feasible_point = res.x[:-1]
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    return ConvexHull(hs.intersections).volume
