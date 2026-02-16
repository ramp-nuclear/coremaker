from math import prod

import hypothesis.strategies as st
from hypothesis import settings
from scipy.linalg import norm as norm2
from scipy.spatial.transform import Rotation

from coremaker.geometries.annulus import Annulus, Ring
from coremaker.geometries.ball import Ball, Circle
from coremaker.geometries.box import Box, Rectangle
from coremaker.geometries.cylinder import FiniteCylinder
from coremaker.geometries.hex import HexPrism, Hexagon
from coremaker.surfaces.cylinder import Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.sphere import Sphere
from coremaker.transform import Transform

medfloats = st.floats(-100, 100, allow_subnormal=False)
posfloats = st.floats(0, 100, allow_subnormal=False, exclude_min=True)
rotations = st.builds(Rotation.from_quat,
                      st.tuples(medfloats, medfloats, medfloats, medfloats).filter(
                          lambda x: norm2(x, ord=2) > 1e-3))
translations = st.builds(Transform, st.tuples(medfloats, medfloats, medfloats))
transforms = st.builds(Transform.from_rotation,
                       st.tuples(medfloats, medfloats, medfloats),
                       rotations,
                       st.booleans())

planes = st.tuples(
    st.tuples(medfloats, medfloats, medfloats).filter(lambda x: norm2(x, ord=2) > 1e-3),
    medfloats,
).map(lambda x: Plane(*x[0], x[1]))

cylinders = st.builds(Cylinder,
                      st.tuples(medfloats, medfloats, medfloats),
                      posfloats,
                      st.tuples(medfloats, medfloats, medfloats).filter(lambda x: norm2(x, ord=2) > 1e-3),
                      inside=st.booleans())
spheres = st.builds(Sphere,
                    st.tuples(medfloats, medfloats, medfloats),
                    posfloats,
                    inside=st.booleans(),
                    )

finitecylinders = st.builds(FiniteCylinder,
                            st.tuples(medfloats, medfloats, medfloats),
                            posfloats,
                            posfloats,
                            st.tuples(medfloats, medfloats, medfloats).filter(
                                lambda x: norm2(x, ord=2) > 1e-3),
                            )

annuli = st.builds(Annulus,
                   st.tuples(medfloats, medfloats, medfloats),
                   st.tuples(st.shared(posfloats, key='annulus'), st.floats(0, 0.999, exclude_min=True)).map(
                       prod),
                   st.shared(posfloats, key='annulus'),
                   posfloats,
                   st.tuples(medfloats, medfloats, medfloats).filter(lambda x: norm2(x, ord=2) > 1e-3),
                   )

boxes = st.builds(Box,
                  st.tuples(medfloats, medfloats, medfloats),
                  st.tuples(posfloats, posfloats, posfloats))

balls = st.builds(Ball,
                  st.tuples(medfloats, medfloats, medfloats),
                  medfloats
                  )
hexprisms = st.builds(HexPrism,
                      st.tuples(medfloats, medfloats, medfloats),
                      posfloats,
                      posfloats)

rectangles = st.builds(Rectangle,
                       st.tuples(medfloats, medfloats),
                       st.tuples(posfloats, posfloats))

circles = st.builds(Circle,
                    st.tuples(medfloats, medfloats),
                    medfloats
                    )

rings = st.builds(Ring, st.tuples(medfloats, medfloats),
                  st.tuples(st.shared(posfloats, key='ring'), st.floats(0, 0.999, exclude_min=True)).map(
                      prod),
                  st.shared(posfloats, key='ring'),
                  )

hexagons = st.builds(Hexagon, st.tuples(medfloats, medfloats), medfloats)

settings.register_profile("fast", max_examples=50, deadline=None)
settings.register_profile("thorough", max_examples=500, deadline=None)

