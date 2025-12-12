# NumPy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import NewType

# SciPy
from scipy.spatial.transform import Rotation

# Types
SO3 = NewType("SO3", NDArray)
Vector3 = NewType("Vector3", NDArray)

def Exp(rot_vec: Vector3) -> SO3:
    try:
        rot_vec = np.squeeze(np.array(rot_vec)).astype(float)
        assert rot_vec.shape == (3,)
    except Exception as e:
        raise ValueError(f"Invalid input `rot_vec`. Could not be cast to a "
                         f"`NDArray(3,)`. Error: {e}.")
    R = Rotation.from_rotvec(rot_vec).as_matrix()
    return R


def Log(R: SO3) -> Vector3:
    if not isinstance(R, np.ndarray):
        raise TypeError("R must be a numpy array.")
    if not R.shape == (3, 3):
        raise ValueError("R must be a 3x3 matrix.")

    rot_vec = Rotation.from_matrix(R.copy()).as_rotvec()
    return rot_vec


def plus_operator(R: SO3, omega: Vector3) -> SO3:
    """
    Performs the `+` operation in the SO(3) group, i.e. `R + omega`.

    The 'plus' operator SO(3) x \R^3 -> SO(3) produces an element of SO(3),
    which is the result of composing the reference element R of SO(3) with a
    small rotation omega vector \R^3.

    Inputs
    - R:     `NDArray(3, 3)` the reference rotation matrix, i.e. an element in
             SO(3).
    - omega: `NDArray(3,)` or `NDArray(3, 1)` or `list` the rotation vector
             lying in the tangent space of SO(3) at R.

    Returns
    - R':    `NDArray(3, 3)` the new rotation matrix after applying the 'plus'
             operator.
    """
    return R @ Exp(omega)


def minus_operator(R1: SO3, R2: SO3) -> Vector3:
    """
    Performs the `-` operation in the SO(3) group, i.e. `R1 - R2`.

    The 'minus' operator SO(3) x SO(3) -> \R^3 is defined as the rotation vector
    that when applied to R2 produces R1.

    Inputs
    - R1:  `NDArray(3, 3)` the first rotation matrix, i.e. an element in SO(3).
    - R2:  `NDArray(3, 3)` the second rotation matrix, i.e. an element in SO(3).

    Returns
    - omega: `NDArray(3,)` the rotation vector that when applied to R2 produces
             R1.
    """
    return Log(R2.T @ R1)
