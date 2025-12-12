# Typing
from __future__ import annotations
from typing import Any, Literal

# Numpy
import numpy as np
from numpy.typing import NDArray

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

def dehomogenized(vectors: NDArray) -> NDArray:
    """
    Dehomogenize vectors stored in matrix (..., d + 1), scaling down by the last
    element of each vector and returning a d-dimensional homogeneous vectors in
    matrix of size (..., d).

    Inputs
    - vectors: `NDArray(..., d + 1)` homogeneous input vectors

    Returns
    - dehomogenized_vectors: `NDArray(..., d)` dehomogenized vectors
    """
    return vectors[..., :-1] / vectors[..., -1:]

def homogenized(vectors: NDArray, fill_value: Any = 1) -> NDArray:
    """
    Homogenize d-dimensional vectors stored in matrix (..., d), returning
    homogeneous vectors in matrix of size (..., d + 1) by appending a `1` to the
    last dimension. If specified, the `fill_value` will be used instead of `1`.

    Inputs
    - vectors: `NDArray(..., d)` input vectors

    Optional Inputs
    - fill_value: `Any` value to fill the last dimension of the vectors. Default
                   is `1`.

    Returns
    - homogenized_vectors: `NDArray(..., d + 1)` homogenized vectors
    """
    return np.concatenate(
        (vectors, np.full(vectors.shape[:-1],
                          fill_value,
                          dtype=vectors.dtype)[..., np.newaxis]),
        axis=-1,
        dtype=vectors.dtype
    )

def normalized(x: NDArray, axis: int = -1, norm: Literal['L1', 'L2'] = 'L2') \
    -> NDArray:
    """
    Normalize an array along the provided axis. By default the normalization is
    perfomed along the last axis.

    Inputs
    - x: `NDArray` input array

    Optional Inputs
    - axis: `int` axis along which to normalize the array. Default is `-1`.
    - norm: `Literal['L1', 'L2']` type of normalization to perform. Default is
            `L2`.

    Returns
    - normalized_x: `NDArray` normalized array
    """
    if np.any(np.linalg.norm(x, axis=axis, keepdims=True) == 0):
        raise ValueError("Error in vector normalization as norm is zero.")
    if norm == 'L1':
        return x / np.linalg.norm(x, ord=1, axis=axis, keepdims=True)
    elif norm == 'L2':
        return x / np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    else:
        raise TypeError(f"Invalid normalization argument '{norm}.' " +
                        f"Choose 'L1' or 'L2'.")

def lower_triangular_to_symmetric(x_flat: NDArray, n: int) -> NDArray:
    """
    Build symmetric matrix or matrices from flattened array(s) representing the
    lower triangular part(s). The function assumes that each input array is of
    dimension `n*(n + 1)/2` and returns a symmetric matrix or matrices of shape
    `(n, n)`.

    Inputs:
    - x_flat: `NDArray(.., n*(n + 1)/2, )` flattened array(s) representing the
               lower triangular part(s) of the output symmetric matrix or
               matrices.
    - n:      `int` dimension of the output symmetric matrix or matrices.

    Returns:
    - X:       `NDArray(..., n, n)` symmetric matrix built from the input array.
    """
    if not x_flat.shape[-1] ==  n * (n + 1) // 2:
        raise ValueError(
            "Input array `x_flat` is not of the correct shape. " +
            f"Expected (..., {n * (n + 1) // 2}), got {x_flat.shape}."
        )
    # Get the lower triangular indices of the matrix
    ii = np.tril_indices(n) # (2, n*(n + 1)/2)

    # Build the symmetric matrix
    X = np.zeros((*x_flat.shape[:-1], n, n)) # (..., n, n)
    X[..., ii[0], ii[1]] = x_flat
    X[..., ii[1], ii[0]] = x_flat
    return X

def extract_lower_triangular(x: NDArray) -> NDArray:
    """
    Flatten matrix or matrices to lower triangular part(s). The function assumes
    that each input matrix is of dimension `(n, n)`. It returns a flattened
    array or arrays of shape `(n*(n + 1)/2, )`.

    Inputs:
    - x: `NDArray(..., n, n)` matrix or matrices.

    Returns:
    - x_flat: `NDArray(..., n*(n + 1)/2, )` flattened array(s) representing the
              lower triangular part(s) of the input matrix or matrices.

    Note: the function does not verify the symmetry of the input matrix or
          matrices. That is, the input matrix or matrices do not need to be
          symmetric.
    """
    # Dimension checks
    n = x.shape[-1]
    if not x.shape[-2:] == (n, n):
        raise ValueError(
            "Input array `x` is not of the correct shape. " +
            f"Expected (..., {n}, {n}), got {x.shape}."
        )

    # Get the lower triangular indices of the matrix
    ii = np.tril_indices(n) # (2, n*(n + 1)/2)

    # Build the flattened array
    x_flat = np.zeros((*x.shape[:-2], n * (n + 1) // 2)) # (..., n*(n + 1)/2)
    x_flat[..., ii[0], ii[1]] = x
    return x_flat

class RuntimeUnreachableError(RuntimeError):
    pass

