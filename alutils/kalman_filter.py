# Typing
from typing import Optional, overload

# Numpy
import numpy as np
from numpy.typing import NDArray

# Scipy
try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Types
State = NDArray
Covariance = NDArray

@overload
def prior_update(x_m: None, P_m: Covariance,
                 A: NDArray, b: None, Q: Covariance) \
                   -> tuple[None, Covariance]: ...

@overload
def prior_update(x_m: State, P_m: Covariance,
                 A: NDArray, b: NDArray, Q: Covariance) \
                    -> tuple[State, Covariance]: ...

def prior_update(x_m: State | None, P_m: Covariance,
                 A: NDArray, b: NDArray | None, Q: Covariance) \
                   -> tuple[State | None, Covariance]:
   """
   Prior update of the Kalman Filter, also referred to as the prediction step.

   The dynamics are described by:
     x_k = A * x_{k-1} + b + v,
     where v is a zero-mean Gaussian RV with covariance Q.

   Inputs
   - x_m: `(n, 1)` or `(n,)` a posteriori estimate of the state,
          i.e. x_{k-1|k-1}.
   - P_m: `(n, n)` a posteriori covariance matrix of the state,
           i.e. P_{k-1|k-1}.
   - A:   `(n, n)` state transition matrix.
   - b:   `(n, 1)` or `(n,)` bias vector.
   - Q:   `(n, n)` process noise covariance matrix.

   Outputs
   - x_p: `(n, 1)` or `(n,)` predicted state estimate at the next time step
          when applying the dynamics, i.e. x_{k|k-1}.
   - P_p: `(n, n)` covariance matrix of the state at the next time step, when
          applying the dynamics, i.e. P_{k|k-1}.
   """

   # Assertions
   if x_m is not None and b is None or x_m is None and b is not None:
       raise ValueError(
           "Either both `x_m` and `b` must be provided or none."
       )

   if not isinstance(P_m, np.ndarray) or not isinstance(A, np.ndarray) or \
      not isinstance(A, np.ndarray) or \
      not P_m.ndim == 2 or not A.ndim == 2 or not Q.ndim == 2:
       raise ValueError(
              "The matrices `P_m`, `A`, `Q` must be 2D NumPy arrays."
        )

   n = A.shape[0]
   if not P_m.shape == (n, n) or not A.shape == (n, n) or not Q.shape == (n, n):
         raise ValueError(
                "The matrices `P_m`, `A`, `Q` must all be NumPy arrays of " +
                "shape (n, n) respectively."
         )

   if x_m is not None:
      if not isinstance(x_m, np.ndarray) or not isinstance(b, np.ndarray) or \
         not (x_m.shape == (n,) or x_m.shape == (n, 1)) or \
         not (b.shape == (n,) or b.shape == (n, 1)):
         raise ValueError(
             "The vectors `x_m` and `b` must both be NumPy arrays of shape " +
             "(n, 1) or (n,)."
         )

   # Prior update
   x_p = None
   if x_m is not None:
       x_p = A @ x_m + b
   P_p = A @ P_m @ A.T + Q

   return x_p, P_p

@overload
def measurement_update(x_p: None, P_p: Covariance,
                       z: None, H: NDArray, R: Covariance,
                       KALMAN_GAIN_FORM: bool = True,
                       JOSEPH_FORM: bool = True,
                       symmetry_tol: float = 1e-8,
                       invertibility_eps: float = 1e-20) \
                         -> tuple[None, Covariance]: ...

@overload
def measurement_update(x_p: State, P_p: Covariance,
                       z: State, H: NDArray, R: Covariance,
                       KALMAN_GAIN_FORM: bool = True,
                       JOSEPH_FORM: bool = True,
                       symmetry_tol: float = 1e-8,
                       invertibility_eps: float = 1e-20) \
                         -> tuple[State, Covariance]: ...

def measurement_update(x_p: State | None, P_p: Covariance,
                       z: NDArray | None, H: NDArray, R: Covariance,
                       KALMAN_GAIN_FORM: Optional[bool] = True,
                       JOSEPH_FORM: Optional[bool] = True,
                       symmetry_tol: Optional[float] = 1e-8,
                       invertibility_eps: Optional[float] = 1e-20) \
                         -> tuple[State | None, Covariance]:
   """
   Measurement update of the Kalman Filter, also referred to as the a posteriori
   update.

   The measurement equation is given by:
     z_k = H * x_k + w,
     where w is a zero-mean Gaussian RV with covariance R.

   Inputs
   - x_p: `(n, 1)` or `(n,)` prior estimate of the state, i.e. x_{k|k-1}.
   - P_p: `(n, n)` prior covariance matrix of the state, i.e. P_{k|k-1}.
   - z:   `(m, 1)` or `(m,)` observed measurement, i.e. z_k.
   - H:   `(m, n)` observation matrix.
   - R:   `(m, m)` measurement noise covariance matrix, symmetric definite
          postive.
   - symmetry_eps: `float` tolerance for checking symmetry of matrices. Default
                    is `1e-8`.
   - invertibility_eps: `float` small tolerance for checking invertibility of
                        matrices. Default is `1e-20`.
   Optional inputs
   - KALMAN_GAIN_FORM: `bool` flag to use the Kalman gain form. Default is
                       `True`.
   - JOSEPH_FORM:      `bool` flag to use the Joseph form for the covariance
                       update. This only works when using the `KALMAN_GAIN_FORM`
                       form and improves numerical stability. Default is
                       `True`.

   Outputs
   - x_m: `(n, 1)` or `(n,)` a posteriori state estimate after employing the
           measurement z_k, i.e. x_{k|k}.
   - P_m: `(n, n)` a posteriori covariance matrix of the state after employing
           the measurement z_k, i.e. x_{k|k}.

   Note: the prior covariance matrix P_p is allowed to be positive semi-
         definite, i.e., it could theoretically be singular. However, the
         measurement noise covariance matrix R must be positive definite, i.e.,
         it must be invertible.
   """

   # Assertions
   if x_p is not None and z is None or x_p is None and z is not None:
       raise ValueError(
           "Either both `x_p` and `z` must be provided or none."
       )

   if not isinstance(P_p, np.ndarray) or not isinstance(R, np.ndarray) or \
      not isinstance(H, np.ndarray) or \
      not P_p.ndim == 2 or not R.ndim == 2 or not H.ndim == 2:
       raise ValueError(
           "The matrices `P_p`, `R` and `H must be 2D NumPy arrays."
        )

   m, n = H.shape
   if not R.shape == (m, m) or not P_p.shape == (n, n):
         raise ValueError(
              "The matrices `P_p`, `R` and `H` must be NumPy arrays of shape " +
              "(n, n), (m, m) and (m, n) respectively."
         )

   if x_p is not None:
      if not isinstance(x_p, np.ndarray) or not isinstance(z, np.ndarray) or \
         not (x_p.shape == (n,) or x_p.shape == (n, 1)) or \
         not (z.shape == (m,) or z.shape == (m, 1)):
         raise ValueError(
             "The vectors `x_p` and `z` must NumPy arrays of shape " +
             "(n, 1) or (n,) and (m, 1) or (m,) respectively."
         )
   x_m = None

   # Assumptions
   if np.abs(P_p.T - P_p).max() > symmetry_tol:
       raise ValueError("The provided prior covariance matrix P_p is not " +
                        "symmetric.")

   if np.abs(R.T - R).max() > symmetry_tol:
       raise ValueError("The provided measurement noise covariance matrix R " +
                        " is not symmetric.")

   if np.linalg.det(R) < invertibility_eps:
       raise ValueError("The provided measurement noise covariance matrix R " +
                        " is not invertible.")

   # Kalman gain form
   if KALMAN_GAIN_FORM:

       # Use Cholesky decomposition if scipy is available
       if SCIPY_AVAILABLE:
           HPHR_and_lower = scipy.linalg.cho_factor(H @ P_p @ H.T + R,
                                                    check_finite=False)
           K = scipy.linalg.cho_solve(HPHR_and_lower, b=(P_p @ H.T).T,
                                      check_finite=False).T
       else:
           K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)

       if x_p is not None:
           x_m = x_p + K @ (z - H @ x_p)

       # Jospeh form (for numerical stability)
       if JOSEPH_FORM:
           P_m = (np.eye(n) - K @ H) @ P_p @ (np.eye(n) - K @ H).T + K @ R @ K.T
       else:
           P_m = (np.eye(n) - K @ H) @ P_p

   # Direct form without computing the Kalman gain
   else:

       # For this form, we need P_p to be invertible
       if np.linalg.det(P_p) < invertibility_eps:
           raise ValueError("The provided prior covariance matrix P_p is not " +
                            "is not invertible.")

       P_p_inv, R_inv = np.linalg.inv(R), np.linalg.inv(P_p)
       P_m = np.linalg.inv(H.T @ R_inv @ H + P_p_inv)

       if x_p is not None:
           x_m = x_p + P_m @ H.T @ R_inv @ (z - H @ x_p)

   return x_m, P_m
