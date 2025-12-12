# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import Optional
import math

# Plotly
try:
    import plotly.graph_objects as go
except ImportError:
    pass

# Shapely
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:
    pass

# Utils
from alutils.decorators import requires_package
from alutils.bbox import BBox
from alutils.loggers import get_logger
logger = get_logger(__name__)

def array_to_latex(
        array: NDArray,
        *,
        decimals: int = 2,
        max_dimension: int | None = None,
    ) -> str:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected input `array` to be of type `NDArray`, " +
                        f"but found `{type(array)}`.")

    # Matrix larger than `max_dimension` or more than 2 dimensions (tensor)
    if max_dimension is not None and \
       any(d > max_dimension for d  in array.shape) or array.ndim > 2:
        return r"$" + (r"\times").join(map(str, array.shape)) + \
               r"\text{ matrix}$"
    # Vector R^n
    if array.ndim == 1:
        vals = [f"{x:.{decimals}f}" for x in array]
        return r"$\begin{pmatrix} " + r" \\ ".join(vals) + \
               r" \end{pmatrix}$"
    # Matrix R^{m x n}
    elif array.ndim == 2:
        rows = [" & ".join(f"{x:.{decimals}f}" for x in row) for row in array]
        return r"$\begin{pmatrix} " + r" \\ ".join(rows) + \
               r" \end{pmatrix}$"
    else:
        raise NotImplementedError(f"This error should not be raised.")

@requires_package("shapely")
def get_2d_boundary(vertices: NDArray, faces: NDArray) -> list[NDArray]:
    """
    Computes the 2D boundary of a 2D mesh defined by its vertices and faces.

    Inputs
    - vertices: `NDArray(N, 2)` the 2D vertices of the mesh.
    - faces: `NDArray(M, 3)` of integers representing the faces of the mesh.

    Returns
    - boundaries: `list[NDArray(N, 2)]` the 2D boundaries of the mesh. Since the
                  mesh can be composed of multiple disjoint parts, the
                  boundaries are returned as a list of 2D vertices.
    """
    assert isinstance(vertices, np.ndarray) and isinstance(faces, np.ndarray)
    if len(vertices) == 0 or len(faces) == 0:
        return []
    assert vertices.ndim == 2 and vertices.shape[-1] == 2
    assert faces.ndim == 2 and faces.shape[-1] == 3 and faces.dtype == np.int32

    triangles = []
    for face in faces:
        triangle = Polygon([vertices[face[i]] for i in range(3)])
        triangles.append(triangle)

    try:
        merged_polygon = unary_union(triangles)
    except Exception as e:
        logger.error(f"Error while computing the 2D boundary of the mesh: {e}")
        return []

    boundaries = []
    if isinstance(merged_polygon, Polygon):
        boundaries.append(np.array(merged_polygon.exterior.coords))
    elif isinstance(merged_polygon, MultiPolygon):
        for polygon in merged_polygon.geoms:
            boundaries.append(np.array(polygon.exterior.coords))

    return boundaries


@requires_package("plotly", "shapely")
def gaussian_1d_traces(
    mu: float, var: float, *values: float, S: int = 100,
    color: Optional[str] = 'cyan') -> list[go.Contour | go.Scatter]:
    """
    Generates the traces for a 1D Gaussian distribution. It plots the PDF.

    The function only draws the contour on a square image of size `H x W`, and
    `S` points are used to draw the ellipses.

    Inputs:
    - mu: `float` the mean of the Gaussian distribution.
    - cov: `float` the variance of the Gaussian distribution.
    - *values: `float` any number of x-values to plot on the PDF.
    - S: `int` the number of points used to draw the PDF.

    Returns:
    - traces: `list[go.Scatter]` the traces of the Gaussian distribution.
    """

    if var <= 0:
        raise ValueError(f"Variance of the Gaussian distribution must be " +
                         f"striclty postive to be plotted, found '{var}'.")

    traces = []

    # PDF of the Gaussian distribution
    if var > 1e-8:
        std = math.sqrt(var)
        x = np.linspace(mu - 3 * std, mu + 3 * std, S) # (S, )
        y = 1/math.sqrt(2 * math.pi * var) * np.exp(-(x-mu)**2 / (2 * var))

        pdf_trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                width=1,
                color=color,
            ),
            name="PDF",
        )
        traces.append(pdf_trace)

    # Mean bar
    mean_trace = go.Scatter(
        x=[mu, mu],
        y=[0, 1/math.sqrt(2 * math.pi * var)],
        mode='lines',
        line=dict(
            color=color,
            dash='dash'
        ),
        name=f"mean {mu}",
    )
    traces.append(mean_trace)

    # Bars for every provided value to plot
    for value in values:
        value_trace = go.Scatter(
            x=[value, value],
            y=[0, 1/math.sqrt(2 * math.pi * var)],
            mode='lines',
            line=dict(
                color='lime',
                dash='dash'
            ),
            name=f"x-value {value}",
        )
        traces.append(value_trace)

    return traces


@requires_package("plotly")
def gaussian_2d_traces(
    mu: NDArray, cov: NDArray,
    output_size: Optional[BBox] = None,
    n_ellipse: int = 100, n_contour: int = 100,
    primary_color: str = "cyan", secondary_color: str = "blue",
    colorscale: str | list = "Viridis"
    ) -> list[go.Contour | go.Scatter]:
    """
    Generates the traces for a 2D Gaussian distribution. It plots the mean, the
    1 standard deviation and the 2 standard deviation ellipses, and the contour
    of the Gaussian distribution.

    The function only draws the contour on a square image of size `H x W`. If
    `output_size` is `None`, i.e., the dimensions `H x W` are unknown, then the
    contour will only be plotted within the 2 standard deviation ellipse.

    Inputs:
    - mu: `NDArray(2,)` the mean of the Gaussian distribution.
    - cov: `NDArray(2, 2)` the covariance matrix of the Gaussian distribution.

    Optional inputs:
    - output_size:     `BBox` defining the size of the output image. If `None`,
                        the contour will only be plotted within the 2 standard
                        deviation ellipse. Default is `None`.
    - n_ellipse:       `int` the number of points used to draw the ellipses.
                       Default is `100`.
    - n_contour:       `int` the number of points used to draw the contour map.
    - primary_color:   `str` the primary color used for drawing the curves.
    - secondary_color: `str` the secondary color used for drawing the curves.
    - colorscale:      `str | list[...]` the colorscale used for drawing the
                        contours.

    Returns:
    - traces: `list[go.Contour | go.Scatter]` the traces of the Gaussian
               distribution.
    """

    assert mu.shape == (2,) and cov.shape == (2, 2)
    if np.linalg.det(cov) < 1e-15:
        logger.warning("The covariance matrix is singular, the Gaussian " +
                       "distribution cannot be plotted.")
        # TODO: IMPROVE
        return [
            go.Scatter(
               x=[mu[0]],
               y=[mu[1]],
               mode="markers",
               marker=dict(
                   size=5,
                   color="red",
                   opacity=0.7,
                   symbol="cross",
               ),
               name="mu",
            )
            ]

    # Covariance matrix
    cov_inv = np.linalg.inv(cov) # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov_inv)

    # Compute the principal vectors of the ellipse x^T COV^{-1} x = 1
    P = eigvecs / np.sqrt(eigvals) # (2, 2)

    # Generate points on the ellipse
    thetas = np.linspace(0, 2 * np.pi, n_ellipse) # (S,) where S=n_ellipse
    cos_sin = np.c_[np.cos(thetas), np.sin(thetas)][..., None] # (S, 2, 1)
    points_1std = mu[:, None] + P @ cos_sin # (S, 2, 1)
    points_2std = mu[:, None] + 2 * P @ cos_sin # (S, 2, 1)
    points_1std = points_1std[..., 0] # (S, 2)
    points_2std = points_2std[..., 0] # (S, 2)

    # Handle the case where the output size is unknown
    size = output_size
    if output_size is None:
        size = BBox.from_points(
            mu + P @ np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])[:, :, None]
        )

    # Compute the contour of the Gaussian distribution
    n = int(math.sqrt(n_contour))
    x = np.linspace(size.x1, size.x2, n) # (n,)
    y = np.linspace(size.y1, size.y2, n) # (n,)
    X, Y = np.meshgrid(x, y) # (S, S, 2)
    XY = np.stack([X, Y], axis=-1)[..., None] # (S, S, 2, 1)
    XY = XY - mu[:, None] # (S, S, 2, 1)
    XY_T = XY.swapaxes(-2, -1) # (S, S, 1, 2)
    Z = XY_T @ cov_inv @ XY # (S, S, 1, 1)
    Z = Z[:, :, 0, 0] # (S, S)

    # Remove points outside of 2 std ellise if output_size is None
    if output_size is None:
        Z = np.where(Z <= 2**2, Z, np.nan)

    gaussian_mu_trace = go.Scatter(
        x=[mu[0]],
        y=[mu[1]],
        mode="markers",
        marker=dict(
            size=5,
            color=primary_color,
            opacity=0.7,
            symbol="cross",
        ),
        name="mu",
    )

    gaussian_ellipse_1std_trace = go.Scatter(
        x=points_1std[:, 0], y= points_1std[:, 1],
        mode="lines",
        line=dict(
            width=1,
            color=secondary_color,
        ),
        name="1 std",
    )

    gaussian_ellipse_2std_trace = go.Scatter(
        x=points_2std[:, 0], y= points_2std[:, 1],
        mode="lines",
        line=dict(
            width=1,
            color=primary_color,
        ),
        name="2 std",
    )

    gaussian_contour_trace = go.Contour(
        x=x, y=y, z=Z,
        contours=dict(coloring='heatmap'),
        colorscale=colorscale,
        showscale=False,
        opacity=0.5,
        name="x^T COV^-1 x",
    )

    return [gaussian_contour_trace, gaussian_mu_trace,
            gaussian_ellipse_1std_trace, gaussian_ellipse_2std_trace]


def bound_values_nice(x: NDArray):
    return np.median(x[x >= np.median(x)])


def bin_to_plot(
    x: NDArray, y: NDArray,
    num_bins: Optional[int] = 100,
    min_x: Optional[float] = 0, max_x: Optional[float | None] = None,
    return_y_cov: Optional[float] = False
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Bin some (x, y) datapoints into bins in order to plot them.

    Inputs
    - x: `NDArray(N)`
    - y: `NDArray(N)` or `NDArray(N, d)`

    Returns
    - mean_x:    `NDArray(num_bins)`
    - mean_y:    `NDArray(num_bins)` or `NDArray(num_bins, d)`
    - std_y:     `NDArray(num_bins)`, `NDArray(num_bins, d)` or \
                 `NDArray(num_bins, d, d)`
    - bin_sizes: `NDArray(num_bins)`

    Optional Inputs
    - num_bins:     `int` number of desired bins. Default is 100.
    - min_x:        `float` the lower bound for the plotted x values. Default is
                    `0`.
    - max_x:        `float` the upper bound for the plotted x values. Default is
                    `None`, which will compute the upper bound via the
                    `bound_75_percent` function.
    - return_y_cov: `bool` if `True`, for multi-dimensional y data (i.e. d>1),
                    the function returns the standard deviation instead of the
                    variance of the y data. Default is `False`.
    """
    # Assertions
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected input data `x` to be of type `NDArray`, " +
                        f"but found `{type(x)}`.")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Expected input data `y` to be of type `NDArray`, " +
                        f"but found {type(y)}.")
    if not (y.ndim == 1 or y.ndim == 2):
        raise ValueError(f"Expected input data `y` have one or two" +
                         f"dimensions, but found shape `{y.shape}`.")
    if return_y_cov and y.ndim == 1:
        raise ValueError(f"Cannot set `return_y_cov` to `True` for one-" +
                         f"dimensional y data.")

    if not return_y_cov and y.ndim > 1:
        raise ValueError(f"Need to set `return_y_cov` to `True` when using " +
                         f"multi-dimensional y data.")

    N = len(x)
    if not x.shape == (N,):
        raise ValueError(f"Expected input data `x` to have shape `(N,)`, " +
                        f"but found `{x.shape}`.")
    if not (y.shape == (N,) or y.ndim == 2 and y.shape == (N, y.shape[-1])) :
        raise ValueError(f"Expected input data `y` to have shape `(N,)` or "
                        f"`(N, d)` with N={N}, but found `{y.shape}`.")

    # Initialize output arrays
    mean_x, mean_y, std_y, bin_sizes = [], [], [], []

    # Upper bound of x values to consider
    if max_x is None:
        max_x = bound_values_nice(x)

    # Compute bins and bin the y values
    for b in range(num_bins):

        # Bin range
        l = min_x + (max_x - min_x) / num_bins * b
        r = min_x + (max_x - min_x) / num_bins * (b+1)
        m_x = (l + r) / 2

        # Select datapoints belonging to the bin
        mask = (x >= l) & (x < r)
        bin_x, bin_y = x[mask], y[mask]

        # Compute the mean and the standard deviation (or covariance)
        if len(bin_y) > 0:
            mean_x.append(m_x)
            mean_y.append(bin_y.mean(axis=0))
            std_y.append(
                np.std(bin_y, axis=0) if not return_y_cov else np.cov(bin_y.T)
            )
            bin_sizes.append(len(bin_x))

    # Return mean_x, mean_y, std_y (or cov_y) and bin_sizes
    return np.array(mean_x), \
           np.array(mean_y), np.array(std_y), \
           np.array(bin_sizes)
