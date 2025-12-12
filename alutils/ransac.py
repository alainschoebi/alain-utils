# Typing
from __future__ import annotations
from typing import Callable, Any, NewType

# Numpy
import numpy as np
from numpy.typing import NDArray

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

Model = NewType("Model", Any)
def ransac(
    model_fct: Callable[[NDArray], Model],
    error_fct: Callable[[NDArray, Model], NDArray],
    data: NDArray,
    n_datapoints: int,
    threshold: float,
    outlier_ratio: float = None,
    n_iterations: int = None
    ) -> tuple[Any, NDArray]:
    """
    RANSAC method finding the best model while rejecting outlier data points.

    Inputs
    - model_fct:    `Callable[[NDArray], Model]` function that finds a model
                     given some data, i.e `model_fct(x: NDArray) -> Model`.
    - error_fct:     `Callable[[NDArray, Model], NDArray] function that computes
                     the error for every datapoint given a model,
                     i.e. `error_fct(x: NDArray, model: Model) -> NDArray`.
    - data:          `NDArray(N, ...)` the N data points used in the problem.
    - n_datapoints:  `int` the minimum number of data points needed to estimate
                     the model.
    - threshold:     `float` threshold value that determines if the model fits a
                     datapoint well or not.
    - outlier_ratio: `float` the estimated outlier ratio. If unspecified or
                     `None`, the `n_iterations is employed.
    - n_iterations:  `int` the number of iterations that will be performed if
                     the `outlier_ratio` is unspecified or `None`. If `None`, an
                     adaptive RANSAC method will be employed.
    Returns
    - best_model:   `Model` the model that fits the data the best
    - inlier_mask:  `NDArray(N, )` boolean array mask for the determined inliers
    """

    if len(data) < n_datapoints:
        logger.error("Fewer datapoints than the size of the subsets needed " +
                      "to estimate the model.")
        raise ValueError("Fewer datapoints than the size of the subsets " +
                         "needed to estimate the model.")

    if outlier_ratio != None:
        prob_success = 0.99
        n_iterations = int(np.ceil(
            np.log(1 - prob_success) /\
            np.log(1 - (1 - outlier_ratio)**n_datapoints)
        ))
    elif n_iterations == None:
        logger.error("Adapative RANSAC is not implemented.")
        raise NotImplementedError("Adapative RANSAC is not implemented.")

    max_n_inliers = 0
    best_model, best_inlier_mask = None, None
    for _ in range(n_iterations):

        # Pick a subset
        subset_idxs = np.random.choice(len(data), size=n_datapoints,
                                       replace=False)
        data_sub = data[subset_idxs]

        # Determine the model, get the error and set the inlier mask
        model = model_fct(data_sub)     # Model
        error = error_fct(data, model)  # (N, )
        inlier_mask = error < threshold # (N, )

        # Count number of inliers and update the best model if necessary
        n_inliers = np.sum(inlier_mask)
        if n_inliers > max_n_inliers:
            max_n_inliers = n_inliers
            best_model, best_inlier_mask = model, inlier_mask

    return best_model, best_inlier_mask
