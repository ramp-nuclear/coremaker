"""Tests for the Transform object

"""
import hypothesis.strategies as st
import numpy as np
from conftest import medfloats, rotations, transforms
from hypothesis import given, settings
from scipy.spatial.transform import Rotation

from coremaker.transform import Transform, identity

arrays = st.tuples(medfloats, medfloats, medfloats).map(np.array)


@settings(max_examples=500)
@given(rotations, arrays)
def test_just_rotation_does_rotation(rot: Rotation, vec: np.ndarray):
    trans = Transform.from_rotation(rotation=rot)
    ours = trans @ vec
    theirs = rot.apply(vec)
    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-5)


@given(st.integers(-100, 100))
def test_rotation_by_2pi_is_multiplication_identity(n: int):
    rot2pi = Transform.from_rotvec(rotation=np.array([0, 0, n * 2 * np.pi]))
    assert identity == rot2pi


@settings(max_examples=500)
@given(arrays, arrays)
def test_just_translation_does_translation(trans: np.ndarray, vec: np.ndarray):
    transf = Transform(translation=trans)
    assert np.allclose(transf @ vec, trans + vec, rtol=1e-5, atol=1e-5)


@settings(max_examples=500)
@given(transforms, arrays)
def test_inv_transform_is_inverse(transform: Transform, vec: np.ndarray):
    inverse = transform.inv()
    one_way = inverse @ transform @ vec
    other_way = transform @ inverse @ vec
    assert np.allclose(one_way, vec, rtol=1e-5, atol=1e-4)
    assert np.allclose(other_way, vec, rtol=1e-5, atol=1e-4)


@settings(max_examples=500)
@given(transforms, transforms, arrays)
def test_transform_matmul_is_associative(t1: Transform, t2: Transform, vec: np.ndarray):
    assert np.allclose((t1 @ t2) @ vec, t1 @ (t2 @ vec), rtol=1e-5, atol=1e-5)


@settings(max_examples=500)
@given(transforms)
def test_translation_has_correct_shape(t: Transform):
    assert t.translation.shape == (3, 1)
