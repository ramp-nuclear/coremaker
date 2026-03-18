"""Utilities for surface objects"""

from functools import partial
from itertools import product
from typing import Iterable

import numpy as np

DECIMAL_PRECISION = 5
PRECISION = 10**-DECIMAL_PRECISION
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


cardinal_rotations = {canonize(_make_rotvec(n * np.pi / 2, ind)) for n, ind in product(range(4), range(3))}
