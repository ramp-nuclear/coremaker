from collections import Counter, defaultdict
from math import isnan
from typing import Dict
import numpy as np

from isotopes import Isotope, ZAID, avogadro
from multipledispatch import dispatch

from coremaker.component import ConcreteComponent as Component
from coremaker.core import Core
from coremaker.protocols.element import Element
from coremaker.tree import Tree


ND = float
cm3 = float
kg = float


def nd_to_kg(isotope: Isotope, nd: ND, volume: cm3):
    return isotope.mass / avogadro * nd * volume / 1000


def _zero(): return 0.


@dispatch(Component)
def parse_amounts(component: Component) -> Dict[Isotope, kg]:
    amounts = Counter()
    volume = component.geometry.volume
    amounts.update(
        {isotope: nd_to_kg(isotope, nd, volume) for
         isotope, nd in component.mixture.isotopes.items() if
         isinstance(isotope, Isotope)})
    return amounts


@dispatch(Tree)
def parse_amounts(element: Element) -> Dict[Isotope, kg]:
    amounts = Counter()
    for component in element.components():
        if component.geometry.volume is not None:
            amounts.update(parse_amounts(component))
    return dict(sorted(amounts.items()))


@dispatch(Tree, Isotope)
def parse_amounts(element: Tree, isotope: Isotope):
    return parse_amounts(element).get(isotope, 0.0)


@dispatch(Core)
def parse_amounts(core: Core) -> Dict[Isotope, kg]:
    """
    return total amounts of isotopes in the core
    """
    amounts = Counter()
    for element in core.all_elements:
        amounts.update(parse_amounts(element))
    return defaultdict(_zero, sorted(amounts.items()))


@dispatch(Core, Isotope)
def parse_amounts(core: Core, isotope: Isotope) -> kg:
    """
    return total amounts of a specific isotope in the core
    """
    return parse_amounts(core).get(isotope, 0.)


@dispatch(Core, ZAID, float)
def parse_amounts(core: Core, isotope: ZAID, mass: float) -> kg:
    """Return total amount of a specific ZAID, given its mass.
    """
    return mass * parse_numeric_amount(core, isotope) / avogadro / 1000


def parse_numeric_amount(core: Core, isotope: ZAID,
                         allow_nan: bool = False) -> float:
    """Parse the numeric amount of an isotope in the core.

    Parameters
    ----------
    core - Core to count in.
    isotope - Isotope to count.
    allow_nan - Whether nan volumes are counted. If False, nan volumes
                are ignored, which gives a numeric amount but maybe a wrong one.

    """
    return sum(vol * comp.mixture.get(isotope, 0.)
               for element in core.all_elements for comp in element.components()
               if not isnan(vol := (comp.geometry.volume or np.nan)) or allow_nan)
