"""Tests for the grid and lattice objects."""

from string import ascii_uppercase

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from coremaker.grids import (
    CartesianGrid,
    CartesianLattice,
    GeneralSpacedGrid,
    HexagonalGrid,
    HexagonalLattice,
    SpacedGrid,
)
from coremaker.grids.cartgrid import cartesian_sites
from coremaker.materials.water import make_light_water

water = make_light_water(20.0)
positivefloats = st.floats(1e-3, 10.0)
positiveints = st.integers(1, 10)
oddpositiveints = positiveints.map(lambda x: 2 * x - 1)
evenpositiveints = positiveints.map(lambda x: 2 * x)
float3d = st.tuples(*(3 * [positivefloats]))
float2d = st.tuples(*(2 * [positivefloats]))
oddint2d = st.tuples(*(2 * [oddpositiveints]))
evenint2d = st.tuples(*(2 * [evenpositiveints]))
oddcartlattices = st.builds(CartesianLattice, float3d, oddint2d, float2d, positivefloats, st.just(water))


@given(oddcartlattices)
def test_cartlattice_at_center_index_is_origin(lat: CartesianLattice):
    idx = tuple(i // 2 for i in lat.shape[::-1])
    assert np.allclose(np.zeros(3), lat.center(idx)), (idx, lat.center(idx))  # type: ignore


alphabet = ascii_uppercase
oddcartgrids = st.builds(CartesianGrid, float3d, oddint2d, float2d, positivefloats, st.just(water))
spacedgrid = st.builds(
    SpacedGrid, float3d, evenint2d, float2d, positivefloats, positivefloats, positivefloats, st.just(water)
)


@given(oddcartgrids)
def test_grid_center_is_at_site_center_in_odd_cases(grid: CartesianGrid):
    shape = grid.lattice.shape
    center = np.array(shape) // 2
    letter = alphabet[center[1]]
    site = f"{letter}{center[0] + 1}"
    assert np.allclose(grid.lattice.center(grid.site_index(site)[1]), np.zeros(3))


@pytest.mark.parametrize(("site", "res"), [("A1", (0, 0)), ("B1", (1, 0)), ("A3", (0, 2)), ("D5", (3, 4))])
def test_known_sites_are_at_known_indices_in_cartesian_grid(site: str, res: tuple[int, int]):
    grid = CartesianGrid((0.0, 0.0, 0.0), (10, 10), (1.0, 1.0), 1.0, water)
    _, calc = grid.site_index(site)
    assert calc == res


@given(spacedgrid)
def test_order_of_lattices_is_correct(grid: SpacedGrid):
    lower_left = grid.lattices[0]
    lower_right = grid.lattices[1]
    upper_left = grid.lattices[2]
    upper_right = grid.lattices[3]
    assert grid.site_index("A1")[0] == lower_left
    assert np.all(lower_left.origin <= lower_right.origin)
    assert np.all(lower_right.origin <= upper_right.origin)
    assert np.all(lower_left.origin <= upper_left.origin)
    assert np.all(upper_left.origin <= upper_right.origin)


shapes = st.builds(lambda x, y: (x, y), positiveints, positiveints)


@given(shapes)
def test_sites_generation(shape):
    sites = set(cartesian_sites(shape))
    assert len(sites) == np.prod(shape)
    for letter in alphabet[: shape[1]]:
        for number in range(1, shape[0] + 1):
            site = f"{letter}{number}"
            assert site in sites


def test_general_spaced_grid_generalizes_spaced_grid():
    general = GeneralSpacedGrid((0.0, 0.0, 0.0), (10, 10), (7.72, 7.72), 200, [5], [5], [2], [2], water)
    special = SpacedGrid((0.0, 0.0, 0.0), (10, 10), (7.72, 7.72), 200, 2, 2, water)
    assert str(general) == str(special)


odd_ascending_int_2d = st.tuples(*(2 * [oddpositiveints])).filter(lambda odd_tuple: odd_tuple[1] <= odd_tuple[0])
hexlattices = st.builds(
    HexagonalLattice, float3d, odd_ascending_int_2d, positivefloats, positivefloats, positivefloats, st.just(water)
)


@given(hexlattices)
def test_hexlattices_at_center_index_is_origin(lat: HexagonalLattice):
    assert np.allclose(tuple(np.zeros(3)), lat.center(tuple(np.zeros(2))))  # type: ignore


hexgrids = st.builds(
    HexagonalGrid, float3d, odd_ascending_int_2d, positivefloats, positivefloats, positivefloats, st.just(water)
)


@given(hexgrids)
def test_hex_grid_center_is_at_site_center(grid: HexagonalGrid):
    shape = grid.lattice.shape
    center = np.array(shape) // 2
    letter = alphabet[center[0]]
    site = f"{letter}{center[1] + 1}"
    assert np.allclose(grid.lattice.center(grid.site_index(site)[1]), np.zeros(3))
