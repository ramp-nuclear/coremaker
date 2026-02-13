"""
Mesh objects, used for mesh based calculations
"""
import numpy as np
from more_itertools import is_sorted
from numpy.typing import ArrayLike


class CartesianMesh:
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

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike):
        self.x = np.asfarray(x)
        self.y = np.asfarray(y)
        self.z = np.asfarray(z)
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

    def __eq__(self, other):
        return all(np.equal(getattr(self, attr), getattr(other, attr)).all()
                   for attr in ['x', 'y', 'z'])

    def __hash__(self):
        return hash(tuple(tuple(getattr(self, attr)) for attr in ['x', 'y', 'z']))


class CylindricalMesh:
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

    def __init__(self, r: ArrayLike, z: ArrayLike, theta: ArrayLike = (0.0, 2 * np.pi)):
        self.r = np.asfarray(r)
        self.z = np.asfarray(z)
        self.theta = np.asfarray(theta)
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


class SphericalMesh:
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

    def __init__(self, r: ArrayLike, theta: ArrayLike | None = None, phi: ArrayLike | None = None):
        self.r = np.asfarray(r)
        self.theta = np.asfarray(theta) if theta else np.array([0, np.pi])
        self.phi = np.asfarray(phi) if phi else np.array([0, 2 * np.pi])
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
