"""Coordinate transformation object. Used for translation and rotation of other objects in 3D space.

"""
from typing import Tuple, TypeVar, Union, Any

import numpy as np
from scipy.sparse import csr_matrix, identity as iden
from scipy.spatial.transform import Rotation

PRECISION = 6
TOLERANCE = 10 ** (-PRECISION)

_zvec = np.zeros(3, dtype=np.float64)
_TransVec = TypeVar('_TransVec',
                    bound=Union["Transform", np.ndarray])
_Triplet = Tuple[float, float, float] | np.ndarray


class Transform:
    r"""Coordinate transformation to translate and rotate objects around.
    Assumes the object is moved and then its location is rotated around the
    3D space.

    This is an Affine transformation.

    The transformation is modeled as a 4x4 matrix, shaped as:

    .. math::
        M = \left( \begin{matrix}
        R & R & R & X \\
        R & R & R & Y \\
        R & R & R & Z \\
        0 & 0 & 0 & 1
        \end{matrix} \right)

    where R are the entries for the rotation matrix in 3D and X, Y, Z are the
    shifts along the x, y and z axes.
    You can verify for yourself that applying this over a column vector
    :math:`\left(\begin{matrix}a\\ b \\ c \\ 1\end{matrix}\right)`
    would yield a shifted and rotated vector as the affine transformation does.

    """
    __slots__ = ['matrix']
    ser_identifier = "Transform"

    def __init__(self, translation: _Triplet = _zvec,
                 rotation: np.ndarray = np.eye(3, dtype=bool),
                 sparse: bool = False,
                 dtype=float):
        """

        Parameters
        ----------
        translation: A triplet of floats or a 3-long array of floats
            A shift of the coordinate center.
        rotation: ArrayLike
            A rotation (3D matrix) of the 3D coordinate space. Shape is (3, 3)
        sparse: bool
            A flag for whether this matrix should be sparse or not.
            If we set this to False by default, and the classmethods have a default True, we get the best speed.
        dtype: DTypeLike
            The dtype to save the matrix as.
        """
        mat = np.empty((4, 4), dtype=dtype)
        mat[:-1, :-1] = rotation
        mat[:-1, -1] = (translation if isinstance(translation, np.ndarray)
                        else np.array(translation, dtype=dtype))
        mat[-1, :] = np.array([0, 0, 0, 1], dtype=dtype)
        self.matrix = csr_matrix(mat) if sparse else mat

    def serialize(self) -> dict[str, Any]:
        if self == identity:
            return {}
        if self.rotation == identity.rotation:
            return {"translation": self.translation.flatten().tolist()}
        if self.is_sparse():
            return {"matrix": self.matrix.toarray().tolist(), "sparse": True}
        return {"matrix": self.matrix.tolist(), "sparse": False}

    @classmethod
    def deserialize(cls, d: dict[str, Any], *_, **__):
        if d == {}:
            return identity
        if "translation" in d:
            return cls(d["translation"], sparse=True)
        mat = d["matrix"]
        return cls.from_matrix(csr_matrix(mat) if d["sparse"] else mat)

    def is_sparse(self) -> bool:
        """Returns whether the underlying matrix is sparse or not.

        """
        return isinstance(self.matrix, csr_matrix)

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "Transform":
        """Create a transform from the 4x4 matrix representation
        
        Parameters
        ----------
        mat: np.ndarry
            The matrix to use as explained in the :class:`Transform` docstring

        Returns
        -------
        Transform
            A new object which is fully described by the given matrix.

        """
        obj = cls.__new__(cls)
        obj.matrix = mat
        return obj

    @classmethod
    def from_rotvec(cls, translation: _Triplet = _zvec,
                    rotation: _Triplet = _zvec,
                    sparse: bool = True,
                    dtype=np.float32) -> "Transform":
        """Build a Transform using Translation and Rotation vectors.

        Parameters
        ----------
        translation: A triplet of floats or a 3-long array of floats
            A shift of the coordinate center.
        rotation: A triplet of floats or a 3-long array of floats
            A rotation vector that points at the axis and its size is the angle.
        sparse: bool
            A flag for whether this matrix should be sparse or not. In most use cases, they are sparse.
        dtype: DTypeLike
            The dtype to save the matrix as.

        """
        return cls(translation=translation,
                   rotation=Rotation.from_rotvec(rotation).as_matrix(),
                   sparse=sparse, dtype=dtype)

    @classmethod
    def from_rotation(cls, translation: _Triplet = _zvec,
                      rotation: Rotation = Rotation.from_rotvec([0., 0., 0.]),
                      sparse: bool = True,
                      dtype=np.float32) -> "Transform":
        """Build a Transform using a Translation vector and a Rotation object.

        Parameters
        ----------
        translation: A triplet of floats or a 3-long array of floats
            A shift of the coordinate center.
        rotation: scipy.spatial.transform.Rotation
            A rotation object from SciPy.
        sparse: bool
            A flag for whether this matrix should be sparse or not. In most use cases, they are sparse.
        dtype: DTypeLike
            The dtype to save the matrix as.

        """
        return cls(translation=translation, rotation=rotation.as_matrix(),
                   sparse=sparse, dtype=dtype)

    @property
    def translation(self) -> np.ndarray:
        """The translation part of the transform

        """
        return self.dense_matrix[:-1, -1].reshape((3, 1))

    @property
    def rotation(self) -> Rotation:
        """Get the rotation part of the transform.

        """
        return Rotation.from_matrix(self.dense_matrix[:-1, :-1])

    @property
    def rotmat(self) -> np.ndarray:
        """The rotation matrix part of the matrix.

        """
        return self.matrix[:-1, :-1]

    @property
    def dense_matrix(self) -> np.ndarray:
        """Get a dense matrix representation of the transformation.

        """
        return np.asarray(self.matrix.todense() if self.is_sparse() else self.matrix)

    @property
    def dtype(self):
        """Get the dtype set for the matrix.

        """
        return self.matrix.dtype

    def __eq__(self, other: "Transform") -> bool:
        try:
            rotation = self.rotation.approx_equal(other.rotation)
            translation = np.allclose(self.translation, other.translation, rtol=1e-10, atol=1e-10)
            return rotation and translation
        except (TypeError, AttributeError):
            return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self.dense_matrix.flatten().round(PRECISION)))

    def __repr__(self) -> str:
        return f'Transform<rotation: {self.rotation.as_rotvec()}, ' \
               f'translation:{self.translation.reshape((3,))}, ' \
               f'sparse: {self.is_sparse()}>'

    def inv(self) -> "Transform":
        """Return the inverse transform of this one, such that applying one
        to the other cancels out.
        """
        inv_rot = self.rotation.inv()
        inv_trans = (-inv_rot.as_matrix() @ self.translation).reshape((3,))
        return type(self).from_rotation(inv_trans, inv_rot)

    def __matmul__(self, other: _TransVec) -> _TransVec:
        try:
            return type(self).from_matrix(self.matrix @ other.matrix)
        except AttributeError:
            other = np.asarray(other)
            vec = np.empty((4,), dtype=other.dtype)
            vec[:-1] = other
            vec[-1] = 1
            return (self.matrix @ vec)[:-1]

    def __bool__(self) -> bool:
        return self != identity


identity = Transform.from_matrix(iden(4, dtype=np.float32, format='csr'))
counterclockwise_90deg = rotate90 = Transform.from_rotvec(rotation=(0., 0., np.pi / 2))
rotate180 = Transform.from_rotvec(rotation=(0., 0., np.pi))
rotate270 = Transform.from_rotvec(rotation=(0., 0., 3 * np.pi / 2))
