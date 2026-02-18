"""Module to create a pyparsing parser for chemical formulae.

"""

from collections import defaultdict
from typing import Sequence

from isotopes import Isotope
from pyparsing import (Suppress, Word, nums, alphas, Forward, Group,
                       Optional, OneOrMore, ParseResults)


__all__ = ['parse_chemical']

_LPAR, _RPAR = map(Suppress, "()")
_integer = Word(nums)
_integer.set_parse_action(lambda t: int(t[0]))  # Parse time conversion to int
_element = Word(alphas.upper(), alphas.lower())

_formula = Forward()
_term = Group((_element | Group(_LPAR + _formula + _RPAR)("subgroup")) +
              Optional(_integer, default=1)("mult"))
_formula << OneOrMore(_term)


def _multiply_contents(tokens):
    """Parse action to multiply out subgroups

    Parameters
    ----------
    tokens:
        Sequence whose first item is the token to parse.

    """
    t = tokens[0]
    # if the token contains a subgroup, then use multiplier to
    # extend counts of all elements in the subgroup
    if t.subgroup:
        mult = t.mult
        for term in t.subgroup:
            term[1] *= mult
        return t.subgroup


_term.set_parse_action(_multiply_contents)


def _sum_by_element(tokens: Sequence[tuple[str, int]]):
    """Parse action to sum up multiple references to the same element

    Parameters
    ----------
    tokens: Sequence[tuple[str, int]
        Sequence of tuples of elements and their number in the partially parsed
        formula

    Returns
    -------

    """
    elements_list = [t[0] for t in tokens]

    duplicates = len(elements_list) > len(set(elements_list))
    if duplicates:
        ctr = defaultdict(int)
        for t in tokens:
            element, num = t[0], t[1]
            ctr[element] += num
        return ParseResults([ParseResults([k, v]) for k, v in ctr.items()])


_formula.set_parse_action(_sum_by_element)


def parse_chemical(s: str) -> dict[Isotope, int]:
    """Parses a chemical formula string into how many atoms of each element
    are present in the compound (i.e. molecule).

    BNF for simple chemical formula (no nesting)

        integer :: '0'..'9'+
        element :: 'A'..'Z' 'a'..'z'*
        term :: element [integer]
        formula :: term+


    BNF for nested chemical formula
        integer :: '0'..'9'+
        element :: 'A'..'Z' 'a'..'z'*
        term :: (element | '(' formula ')') [integer]
        formula :: term+


    Parameters
    ----------
    s: str
        chemical formula

    Returns
    -------
    dict[Isotope, int]
        A dictionary of how many times each element (as an Isotope) appear in
        the chemical formula.

    """
    return dict((Isotope.from_name(x), n) for x, n in _formula.parse_string(s).asList())
