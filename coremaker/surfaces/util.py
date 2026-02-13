"""Utilities for surface objects

"""
from functools import partial
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

from coremaker.surfaces.plane import Plane

DECIMAL_PRECISION = 5
PRECISION = 10 ** -DECIMAL_PRECISION
isclose = partial(np.isclose, atol=10 * PRECISION, rtol=10 * PRECISION)
allclose = partial(np.allclose, atol=10 * PRECISION, rtol=10 * PRECISION)


def comma_format(it: Iterable[float]) -> str:
    """A string representation of float iterables

    Parameters
    ----------
    it: Iterable[float]
        The iterable of floats to format.

    """
    return f"({','.join(f'{v:.3e}' for v in it)})"


def canonize(v: np.ndarray) -> tuple[str, ...]:
    """Turn an array of angle-related floats into a tuple of strings in canon format.

    This allows for the use of practically immutable arrays as keys in a set or dict.
    Notice that the angle-relation comes from negative values being replaced by 2*pi+v
    to ensure positive canon form.

    Specifically, when you only want to allow certain specific array values,
    like in cardinal rotations, you want to make sure they are in an allowed set.

    Parameters
    ----------
    v: np.ndarray
        The array to turn into cannon format.

    Returns
    -------
    A tuple of strings, for each element in the array.

    Examples
    --------
    >>> a = np.array([1., 3.322, -2.5, -0.1, 0.])
    >>> canonize(a)
    ('1.e+00', '3.322e+00', '3.78319e+00', '6.18319e+00', '6.28319e+00')

    """
    return tuple(
        np.format_float_scientific(vv, DECIMAL_PRECISION)
        if vv > -0.0
        else np.format_float_scientific(2 * np.pi + vv, DECIMAL_PRECISION)
        for vv in v
    )


def _make_rotvec(rot: float, i: int) -> np.ndarray:
    arr = np.zeros(3, dtype=np.float32)
    arr[i] = rot
    return arr


cardinal_rotations = {
    canonize(_make_rotvec(n * np.pi / 2, ind)) for n, ind in product(range(4), range(3))
}


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
