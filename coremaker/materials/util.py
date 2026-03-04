"""Utilities for common materials

"""
from typing import Any, Iterable

from coremaker.materials._chemical_parser import parse_chemical

room_temperature = 20.  # deg C.
__all__ = ['cumulative_dict', 'parse_chemical', 'room_temperature']


def cumulative_dict(args: Iterable[tuple[Any, float]],
                    **kwargs: float) -> dict[Any, float]:
    """Creates a dictionary where repeating keys are summed in value

    The parameters follow the typing of a dictionary whose values are floats.
    """
    d = dict()
    for key, v in args:
        d[key] = v + d.get(key, 0.)
    for key, v in kwargs.items():
        d[key] = v + d.get(key, 0.)
    return d
