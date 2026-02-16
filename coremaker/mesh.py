"""
Mesh objects, used for mesh based calculations
"""
from typing import Type, TypeVar

from coremaker.elements.util import split

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")

import numpy as np
from more_itertools import is_sorted
from numpy.typing import ArrayLike
from ramp_core.serializable import Serializable

from coremaker.units import cm


class CartesianMesh(Serializable):
    """
    Represents a cartesian mesh in x,y,z

    Parameters
    ----------
    x : ArrayLike
      the x coordinates of the mesh in increasing order.
    y : ArrayLike
      the y coordinates of the mesh in increasing order.
    z : ArrayLike
      the z coordinates of the mesh in increasing order.
    """

    ser_identifier = "CartMesh"

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike):
        self.x, self.y, self.z = (np.asarray(v, dtype=float) for v in (x, y, z))
        if len(self.x) <= 1:
            raise ValueError('length of x must be >=2')
        if len(self.y) <= 1:
            raise ValueError('length of y must be >=2')
        if len(self.z) <= 1:
            raise ValueError('length of z must be >=2')
        if not is_sorted(self.x):
            raise ValueError('x values must be increasing')
        if not is_sorted(self.y):
            raise ValueError('y values must be increasing')
        if not is_sorted(self.z):
            raise ValueError('z values must be increasing')

    @classmethod
    def from_vertices(cls: Type[Self],
                      lower_left: np.ndarray,
                      upper_right: np.ndarray,
                      resolution: tuple[cm, cm, cm],
                      ) -> Self:
        """Construct a regular mesh based on its bounding vertices and a resolution.

        Parameters
        ----------
        lower_left: np.ndarray
            Vertex at the lower left of the box, as in Box.from_vertices
        upper_right: np.ndarray
            Vertex at the upper right of the box, as in Box.from_vertices
        resolution: tuple[cm, cm, cm]
            Resolution in each dimension. This is the largest allowed size of a cell in each dimension.

        """
        x, y, z = (np.linspace(lower_left[i], upper_right[i], num=splits+1)
                   for i, splits in enumerate(split(upper_right - lower_left, resolution)))
        return cls(x, y, z)

    def __eq__(self, other):
        return all(np.equal(getattr(self, attr), getattr(other, attr)).all()
                   for attr in ['x', 'y', 'z'])

    def __hash__(self):
        return hash(tuple(tuple(getattr(self, attr)) for attr in ['x', 'y', 'z']))


class CylindricalMesh(Serializable):
    """
    Represents a cylindrical mesh

    Parameters
    ----------
    r : ArrayLike
      the r coordinates of the mesh in increasing order.
    theta : ArrayLike
      the theta coordinates of the mesh in increasing order.
    z : ArrayLike
      the z coordinates of the mesh in increasing order.
    Notes
    -----
    theta must start in 0 and ends with 2pi
    r must start with 0
    """

    ser_identifier = "CylMesh"

    def __init__(self, r: ArrayLike, z: ArrayLike, theta: ArrayLike = (0.0, 2 * np.pi)):
        self.r, self.z, self.theta = (np.asarray(x, dtype=float) for x in (r, z, theta))
        if self.r[0] != 0.0:
            raise ValueError('r values must start with 0')
        if self.theta[0] != 0.0:
            raise ValueError('theta values must start with 0')
        if self.theta[-1] != 2 * np.pi:
            raise ValueError('theta values must end with 2pi')
        if len(self.r) <= 1:
            raise ValueError('length of r must be >=2')
        if len(self.theta) <= 1:
            raise ValueError('length of theta must be >=2')
        if len(self.z) <= 1:
            raise ValueError('length of z must be >=2')
        if not is_sorted(self.r):
            raise ValueError('r values must be increasing')
        if not is_sorted(self.theta):
            raise ValueError('theta values must be increasing')
        if not is_sorted(self.z):
            raise ValueError('z values must be increasing')

    def __eq__(self, other):
        return all(np.equal(getattr(self, attr), getattr(other, attr)).all()
                   for attr in ['r', 'theta', 'z'])


class SphericalMesh(Serializable):
    """
    Represents a spherical mesh

    Parameters
    ----------
    r : ArrayLike
      the r coordinates of the mesh in increasing order.
    theta : ArrayLike
      the theta coordinates of the mesh in increasing order.
    phi : ArrayLike
      the phi coordinates of the mesh in increasing order.
    Notes
    -----
    theta must start in 0 and ends with pi
    phi must start in 0 and ends with 2pi
    r must start with 0
    """

    ser_identifier = "SphereMesh"

    def __init__(self, r: ArrayLike, theta: ArrayLike | None = None, phi: ArrayLike | None = None):
        self.r = np.asarray(r, dtype=float)
        self.theta = np.asarray(theta, dtype=float) if theta is not None else np.array([0, np.pi])
        self.phi = np.asarray(phi, dtype=float) if phi is not None else np.array([0, 2 * np.pi])
        if self.r[0] != 0.0:
            raise ValueError('r values must start with 0')
        if self.theta[0] != 0.0:
            raise ValueError('theta values must start with 0')
        if self.theta[-1] != np.pi:
            raise ValueError('theta values must end with pi')
        if self.phi[0] != 0.0:
            raise ValueError('phi values must start with 0')
        if self.phi[-1] != 2 * np.pi:
            raise ValueError('phi values must end with 2pi')
        if len(self.r) <= 1:
            raise ValueError('length of r must be >=2')
        if len(self.theta) <= 1:
            raise ValueError('length of theta must be >=2')
        if len(self.phi) <= 1:
            raise ValueError('length of phi must be >=2')
        if not is_sorted(self.r):
            raise ValueError('r values must be increasing')
        if not is_sorted(self.theta):
            raise ValueError('theta values must be increasing')
        if not is_sorted(self.phi):
            raise ValueError('phi values must be increasing')

    def __eq__(self, other):
        return all(np.equal(getattr(self, attr), getattr(other, attr)).all()
                   for attr in ['r', 'theta', 'phi'])

jsonable = [CartesianMesh, CylindricalMesh, SphericalMesh]
Mesh = CartesianMesh | CylindricalMesh | SphericalMesh
