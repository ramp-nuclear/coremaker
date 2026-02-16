"""Tests for the Mixture object.

"""
import re
from collections import Counter
from typing import Dict, Tuple

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from isotopes import ZAID, He4, H, Al27, Al, Si, C, O, Cr, He, Isotope

from coremaker.materials import Mixture
from coremaker.materials.gases import ideal_gas_mixture
from coremaker.materials.util import parse_chemical
from coremaker.materials.water import make_light_water
from coremaker.protocols.mixture import Chemical, are_close

protons = st.integers(min_value=0, max_value=100)
nucleons = st.integers(min_value=0, max_value=400)
excites = st.integers(min_value=0, max_value=5)
isos = st.builds(ZAID, protons, nucleons, excites)
temperatures = st.floats(min_value=0., max_value=400.)
densities = st.floats(min_value=1e-6, max_value=2.)
isodicts = st.shared(st.dictionaries(isos, densities, min_size=1), key='dict')
keys = isodicts.flatmap(lambda d: st.sampled_from(tuple(d.keys())))


@pytest.mark.parametrize("chem", list(Chemical))
def test_chemical_has_associated_isotopes(chem):
    assert chem.isotopes


@given(isodicts, keys)
def test_mixture_closeness_with_small_density_change(
        isodict: dict[ZAID, float], key: ZAID):
    iso2 = {i: (1 + 1e-16 if i == key else 1) * v for i, v in isodict.items()}
    mixture_1 = Mixture(isodict, 20)
    mixture_2 = Mixture(iso2, 20)
    assert are_close(mixture_1, mixture_2)


zaids = (st.tuples(st.integers(min_value=0),
                   st.integers(min_value=0, max_value=998),
                   st.integers(min_value=0, max_value=9))
         .map(lambda x: ZAID(x[0], x[1], x[2])))
number_densities = st.floats(min_value=1e-10, max_value=1.,
                             allow_nan=False,
                             allow_infinity=False)
isotope_dicts = st.dictionaries(keys=zaids, values=number_densities, min_size=1)

temp_pair = st.tuples(st.floats(), st.floats()).filter(lambda x: abs(x[0] - x[1]) > 1)


@given(isotope_dicts, temp_pair)
def test_mixture_inequality_when_temperature_is_different(
        isotopes: Dict[ZAID, float], temps: Tuple[float, float]):
    t1, t2 = temps
    mixture_1 = Mixture(isotopes, t1)
    mixture_2 = Mixture(isotopes, t2)
    assert (mixture_1 != mixture_2)


chemicals = st.sampled_from(Chemical)
chemlists = st.shared(st.lists(chemicals), key="moo")
chemical_shuffled_list = chemlists.flatmap(lambda x: st.permutations(x))


@given(isotope_dicts,
       st.floats(allow_nan=False),
       chemical_shuffled_list,
       chemical_shuffled_list)
def test_order_of_chemicals_does_not_matter(isotopes, temp, c1, c2):
    mix1 = Mixture(isotopes, temp, c1)
    mix2 = Mixture(isotopes, temp, c2)
    assert mix1 == mix2


shared_dicts = st.shared(st.dictionaries(keys=zaids, values=number_densities, min_size=1),
                         key='dictmoo')
perm_dicts = (shared_dicts.flatmap(lambda x: st.permutations(list(x.items())))
              .map(lambda x: dict(x)))


@given(perm_dicts, perm_dicts, st.floats(allow_nan=False))
def test_order_of_isotopes_does_not_matter(iso1, iso2, t):
    mix1 = Mixture(iso1, t)
    mix2 = Mixture(iso2, t)
    assert mix1 == mix2


# noinspection PyPep8Naming
def test_room_condition_helium_gas_has_low_but_existing_density_He4():
    helium = {He4: 1}
    gas = ideal_gas_mixture(helium, 1., 20.)
    assert set(gas.keys()) == {He4}
    assert 1e-20 < gas[He4] < 1e-4


def test_light_water_at_20degc_by_regression(num_regression):
    water = make_light_water(20.)
    isotopes_d = {str(iso): den for iso, den in water.isotopes.items()}
    num_regression.check(isotopes_d)


def test_example_creation_of_mixture_does_not_throw():
    Mixture({H: 1}, 20)


def test_expand_cummulative_with_both_isotope_and_element_by_simple_example():
    mix = Mixture({Al27: 1.0, Al: 1.0}, 20)
    assert Mixture.expand(mix) == Mixture({Al27: 2.0}, 20)


def test_expand_elements_expands_just_the_required_elements_by_example():
    mix = Mixture({Si: 1.0, Al: 1.0}, 20)
    assert Mixture.expand(mix, elements=(Al,)) == Mixture({Al27: 1.0, Si: 1.0}, 20)


def test_alloy_of_something_with_itself_is_the_same_as_just_its_by_weight_with_example():
    assert Mixture.alloy(Al27, {Al27: 0.5}, 1, 20) == Mixture.by_weight_density({Al27: 1.0}, 20)


def _dictionary_to_group(d: dict[str, int], squeeze=True) -> str:
    prefix, suffix = ('()' if not squeeze or len(d) > 1 else ('', ''))
    content = ''.join((f'{key}{val}' for key, val in d.items()))
    return f'{prefix}{content}{suffix}'


_possible_elements = [H, C, O, Cr, He]
elements = st.sampled_from(_possible_elements)
element_names = elements.map(lambda x: x.symbol)
nums = st.integers(min_value=1, max_value=5)
group_dicts = st.dictionaries(element_names, nums, min_size=1)
groups = group_dicts.map(_dictionary_to_group)
formulae = st.dictionaries(groups, nums, min_size=1).map(_dictionary_to_group)
compositions = st.dictionaries(formulae, number_densities)
direct_dicts = st.dictionaries(elements, number_densities)


def extend_tuple(strategy):
    return st.dictionaries(strategy, nums, min_size=1).map(lambda x: tuple(x.items()))


recursive_tuple = st.recursive(st.dictionaries(element_names, nums, min_size=1).map(lambda x: tuple(x.items())),
                               extend_tuple, max_leaves=20)


def _recursive_tuple_to_str(x):
    if isinstance(x, str):
        return x
    return _dictionary_to_group({_recursive_tuple_to_str(key): value for key, value in x}, squeeze=False)


def _recursive_tuple_to_dictionary(x):
    def flatten_tup(tup):
        if isinstance(tup, str):
            return tup
        return ''.join(flatten_tup(key) * value for key, value in tup)

    return {Isotope.from_name(key): value
            for key, value in Counter(re.findall('[A-Z][a-z]?', flatten_tup(x))).items()}


@settings(deadline=None)
@given(recursive_tuple)
def test_chemical_parse_of_recursive_formula(tup):
    recursive_formula = _recursive_tuple_to_str(tup)
    assert parse_chemical(recursive_formula) == _recursive_tuple_to_dictionary(tup)


@given(group_dicts)
def test_chemical_parse_of_single_group_inverse_of_dict_to_group(d):
    nd = parse_chemical(_dictionary_to_group(d))
    assert {key.symbol: v for key, v in nd.items()} == d


@given(compositions, direct_dicts)
def test_chemical_parsing_mixtures_succeeds_and_adds_up(forms, isod):
    d = forms | isod
    mix = Mixture.by_chemical_formula(d, 20.)
    for iso, val in isod.items():
        assert mix[iso] >= val
    if forms:
        assert mix.isotopes != isod
