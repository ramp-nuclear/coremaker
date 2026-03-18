import json
from collections import Counter
from pathlib import PurePath
from string import ascii_lowercase

import hypothesis.strategies as st
import numpy as np
from conftest import (
    annuli,
    balls,
    circles,
    cylinders,
    finitecylinders,
    hexagons,
    hexprisms,
    medfloats,
    planes,
    posfloats,
    rectangles,
    rings,
    spheres,
)
from conftest import boxes as boxgeos
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from isotopes import ZAID
from ramp_core import RampJSONDecoder, RampJSONEncoder

from coremaker import Core, jsonable
from coremaker.elements.box import ExcludeFrame
from coremaker.elements.cylindrically_symmetric import AnnulusTree, ChunkedAnnulusTree
from coremaker.geometries import (
    Annulus,
    Ball,
    BareGeometry,
    Circle,
    ConcreteUnionGeometry,
    FiniteCylinder,
    Hexagon,
    HexPrism,
    Rectangle,
    Ring,
    infiniteGeometry,
)
from coremaker.geometries import (
    Box as BoxGeo,
)
from coremaker.geometries.holed import ConcreteHoledGeometry
from coremaker.grids import (
    CartesianGrid,
    CartesianLattice,
    GeneralSpacedGrid,
    HexagonalGrid,
    HexagonalLattice,
    NullGrid,
    SpacedGrid,
)
from coremaker.materials.aluminium import al1050, al6061
from coremaker.materials.mixture import Chemical, Mixture
from coremaker.mesh import CartesianMesh, CylindricalMesh, SphericalMesh
from coremaker.surfaces import Cylinder, Plane, Sphere
from coremaker.tree import Node, Tree

zaids = st.tuples(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=3),
).map(lambda x: ZAID(x[0], x[1], x[2]))
contents = st.dictionaries(zaids, st.floats(min_value=1e-10, max_value=1))
temperatures = st.floats(min_value=5, max_value=70)
sabs = st.lists(st.sampled_from(Chemical)).map(tuple)
mixtures = st.builds(Mixture, contents, temperatures, sab_tables=sabs)

surfaces = st.lists(st.one_of(planes, spheres, cylinders), min_size=1, max_size=4)
bares = surfaces.map(BareGeometry)
geoms = st.one_of(bares, balls, boxgeos, circles, rings, finitecylinders, rectangles, hexprisms, hexagons)
geomseq = st.lists(geoms).map(tuple)
holed = st.builds(ConcreteHoledGeometry, geoms, geomseq, geomseq)
unions = st.builds(
    ConcreteUnionGeometry,
    geomseq.filter(lambda x: len(x) > 1),
    st.one_of(st.none(), st.floats(min_value=1e-5, max_value=1)),
)
centers = st.tuples(*(3 * [medfloats]))
smallposint = st.integers(min_value=1, max_value=10)
gshape = st.tuples(smallposint, smallposint)
dim2d = st.tuples(posfloats, posfloats)
cartlats = st.builds(
    CartesianLattice,
    center=centers,
    shape=gshape,
    dimensions=dim2d,
    height=st.none() | posfloats,
    mixture=mixtures,
)
hexlats = st.builds(
    HexagonalLattice,
    center=centers,
    shape=gshape,
    pitch=posfloats,
    height=st.none() | posfloats,
    outer_radius=posfloats,
    mixture=mixtures,
)

sorted_vec = arrays(
    dtype=float,
    shape=st.integers(min_value=2, max_value=10),
    elements=medfloats,
    unique=True,
).map(np.sort)
cartmesh = st.builds(CartesianMesh, x=sorted_vec, y=sorted_vec, z=sorted_vec)
rad_vec = (
    arrays(
        dtype=float,
        shape=st.integers(min_value=2, max_value=10),
        elements=posfloats,
        unique=True,
    )
    .map(np.sort)
    .map(lambda x: np.concatenate(([0], x)))
)
tau_vec = (
    arrays(
        dtype=float,
        shape=st.integers(min_value=2, max_value=10),
        elements=st.floats(min_value=1e-10, max_value=2 * np.pi - 1e-10),
        unique=True,
    )
    .map(np.sort)
    .map(lambda x: np.concatenate(([0], x, [2 * np.pi])))
)
pi_vec = tau_vec.map(lambda x: x / 2)
cylmesh = st.builds(CylindricalMesh, r=rad_vec, z=sorted_vec, theta=tau_vec)
sphmesh = st.builds(SphericalMesh, r=rad_vec, phi=tau_vec, theta=pi_vec)

nodes = st.builds(Node, geometry=geoms, mixture=st.none() | mixtures)

tree1 = Tree()
tree2 = AnnulusTree(1, 2, 3, al1050, PurePath("root"), (1, 0, 0))
tree3 = ChunkedAnnulusTree(1, 2, 3, al1050, PurePath("root"), (1, 0, 0), (0.1, 0.4))
tree4 = ExcludeFrame(
    frame_dimensions=(4, 4, 4),
    picture_dimensions=(2, 2, 2),
    frame_name=PurePath("frame"),
    picture_name=PurePath("picture"),
    frame_mixture=al1050,
    picture_mixture=al6061,
)
lattice_tree = Tree()
lattice_tree.nodes[PurePath("lat")] = CartesianLattice((0, 0, 0), (2, 2), (5, 5), None, al1050)
example_trees = (tree1, tree2, tree3, tree4, lattice_tree)
trees = st.sampled_from(example_trees)
grid_contents = st.dictionaries(st.text(), trees)

nullgrid = st.just(NullGrid())
cartgrids = st.tuples(grid_contents, cartlats).map(lambda x: CartesianGrid.from_lattice(x[0], x[1]))
spacegrids = st.builds(
    SpacedGrid,
    center=centers,
    shape=gshape.map(lambda x: (2 * x[0], 2 * x[1])),
    lattice_dimensions=st.tuples(posfloats, posfloats),
    height=st.none() | posfloats,
    space_dx=posfloats,
    space_dy=posfloats,
    mixture=mixtures,
    rod_contents=st.none() | grid_contents,
)

holes = st.integers(min_value=0, max_value=5)
holes_width = holes.flatmap(
    lambda x: st.lists(st.tuples(st.integers(min_value=-5, max_value=5), posfloats), min_size=x, max_size=x)
)
hx, hy = st.shared(holes_width, key="x"), st.shared(holes_width, key="y")
xholes, yholes = hx.map(lambda x: [v[0] for v in x]), hy.map(lambda x: [v[0] for v in x])
xdx, ydy = hx.map(lambda x: [v[1] for v in x]), hy.map(lambda x: [v[1] for v in x])
genspacegrids = st.builds(
    GeneralSpacedGrid,
    center=centers,
    shape=gshape,
    lattice_dimensions=dim2d,
    height=st.none() | posfloats,
    holes_x=xholes,
    holes_y=yholes,
    spaces_dx=xdx,
    spaces_dy=ydy,
    mixture=mixtures,
    rod_contents=st.none() | grid_contents,
)

hexgrids = st.tuples(st.just({}) | grid_contents, hexlats).map(lambda x: HexagonalGrid.from_lattice(*x))

grids = st.one_of(nullgrid, cartgrids, spacegrids, genspacegrids, hexgrids)
outers = st.one_of(boxgeos, spheres, finitecylinders, st.none())
shared_trees = st.shared(trees, key="treeshare")
tree_aliases = shared_trees.filter(lambda x: x.nodes).flatmap(
    lambda t: st.dictionaries(
        st.text(ascii_lowercase, min_size=1),
        st.tuples(
            st.just("Explanation"),
            st.lists(st.sampled_from(list(t.nodes.keys())), min_size=1),
        ),
    )
)
cores = st.builds(Core, grid=grids, tree=shared_trees, aliases=tree_aliases, outer_geometry=outers)

strats = {
    Mixture: mixtures,
    Sphere: spheres,
    Plane: planes,
    Cylinder: cylinders,
    type(infiniteGeometry): st.just(infiniteGeometry),
    BareGeometry: bares,
    Ring: rings,
    Circle: circles,
    Rectangle: rectangles,
    Hexagon: hexagons,
    Annulus: annuli,
    Ball: balls,
    BoxGeo: boxgeos,
    FiniteCylinder: finitecylinders,
    HexPrism: hexprisms,
    ConcreteHoledGeometry: holed,
    ConcreteUnionGeometry: unions,
    CartesianLattice: cartlats,
    HexagonalLattice: hexlats,
    CartesianMesh: cartmesh,
    CylindricalMesh: cylmesh,
    SphericalMesh: sphmesh,
    Node: nodes,
    Tree: trees,
    NullGrid: nullgrid,
    CartesianGrid: cartgrids,
    HexagonalGrid: hexgrids,
    SpacedGrid: spacegrids,
    GeneralSpacedGrid: genspacegrids,
    Core: cores,
}


def test_no_two_identifiers_the_same():
    c = dict(Counter([c.ser_identifier for c in jsonable]))
    assert set(c.values()) == {1}, {key: value for key, value in c.items() if value != 1}


def test_strat_for_all_supported():
    assert set(strats) == set(jsonable), (set(jsonable) - set(strats), (set(strats) - set(jsonable)))


RampJSONDecoder.supported = {c.ser_identifier: c for c in jsonable}


def _test_ser_deser(x):
    s = json.dumps(x, cls=RampJSONEncoder)
    try:
        v = json.loads(s, cls=RampJSONDecoder)
    except (RuntimeError, TypeError):
        print(s)
        raise
    assert x == v, (x, v, s)


for cls, strat in strats.items():
    globals()[f"test_ser_deser_{cls.__name__}"] = settings(deadline=None)(given(strat)(_test_ser_deser))
