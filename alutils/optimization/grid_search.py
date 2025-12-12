# Typing
from typing import Callable, TypeVar, cast

# NumPy
import numpy as np
from numpy.typing import NDArray

# Python
from tqdm import tqdm

# Logging
from alutils import get_logger
logger = get_logger(__name__)

# Structures
Point = TypeVar("Point")

class AllPointsInfeasibleError(Exception):
    """
    Custom error to be raised when all points in the grid search are infeasible.
    """
    pass

def grid_search_min(
    points: list[NDArray],
    point_transform: Callable[[NDArray], Point | None],
    feasible: Callable[[Point], bool],
    cost: Callable[[Point], float],
) -> tuple[Point, float, np.ma.MaskedArray]:
    """
    Perform a grid search over the given points in the search for a minimum.

    Inputs
    - points:          `list[NDArray]` list of points to search over. Each point
                       array is a 1D array of coordinates.
    - point_transform: `Callable[[NDArray], Point | None]` function to transform
                       the point coordinates into the desired format. The
                       function should return None if the point is infeasible.
    - feasible:        `Callable[[Point], bool]` function to check if the point
                       is feasible. It should return True if the point is
                       feasible and False otherwise.
    - cost:            `Callable[[Point], float]` function to compute the cost
                       associated to the provided point.
    Returns
    - min_point: `Point` the point with the minimum cost.
    - min_cost:  `float` the minimum cost associated to the point.
    - costs:     `np.ma.MaskedArray` the grid of costs. The masked values
                 represent the infeasible points.

    Raises
    - AllPointsInfeasibleError: if all points in the grid search are infeasible.
    """
    # Grid for storing costs
    costs: np.ma.MaskedArray = np.ma.zeros([len(p) for p in points])

    # Point from index function
    def point_from_index(index: tuple[int, ...]) -> Point | None:
        point = np.array([points[i][index[i]] for i in range(len(points))])
        return point_transform(point)

    # Iterate over all possible points
    pbar = tqdm(np.ndindex(costs.shape), total=np.prod(costs.shape),
                desc="Grid Search", unit="point")
    for index in pbar:
        # Get corresponding point
        point = point_from_index(index)

        # Compute cost if feasible, else mask it
        if point is None or not feasible(point):
            costs[index] = np.ma.masked
        else:
            costs[index] = cost(point)

    # Verify that some points are feasible
    if np.all(costs.mask == True):
        raise AllPointsInfeasibleError("All points are infeasible.")

    # Find the index of the minimum value in the grid
    min_index = np.unravel_index(np.ma.argmin(costs), costs.shape)
    min_point = point_from_index(min_index)
    min_cost = costs[min_index]

    # Log the results
    logger.debug(f"Grid search successful with minimum cost {min_cost:.2f}.")
    logger.debug(
        f"{(~costs.mask).sum()} points out of {costs.size} were " +
        f"feasible ({(~costs.mask).sum()/costs.size*100:.1f}%)."
    )

    # Return the minimum point, its cost and the costs grid
    return cast(Point, min_point), min_cost, costs

def grid_search_max(
    points: list[NDArray],
    point_transform: Callable[[NDArray], Point | None],
    feasible: Callable[[Point], bool],
    cost: Callable[[Point], float],
) -> tuple[Point, float, np.ma.MaskedArray]:
    """
    Perform a grid search over the given points in the search for a maximum.

    Inputs
    - points:          `list[NDArray]` list of points to search over. Each point
                       array is a 1D array of coordinates.
    - point_transform: `Callable[[NDArray], Point | None]` function to transform
                       the point coordinates into the desired format. The
                       function should return None if the point is infeasible.
    - feasible:        `Callable[[Point], bool]` function to check if the point
                       is feasible. It should return True if the point is
                       feasible and False otherwise.
    - cost:            `Callable[[Point], float]` function to compute the cost
                       associated to the provided point.
    Returns
    - max_point: `Point` the point with the maximum cost.
    - max_cost:  `float` the maximum cost associated to the point.
    - costs:     `np.ma.MaskedArray` the grid of costs. The masked values
                 represent the infeasible points.
    """

    min_point, min_cost, costs = grid_search_min(
        points=points,
        point_transform=point_transform,
        feasible=feasible,
        cost=lambda point: -cost(point),
    )

    return min_point, -min_cost, -costs
