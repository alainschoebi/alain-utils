# Typing
from __future__ import annotations
from typing import Optional, Any, cast

# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from pathlib import Path

# Utils
from .decorators import requires_package

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Cython BBox
try:
    import cython_bbox
except ImportError:
    pass

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

class BBox:
    """
    Bounding box class for 2D space.

    The BBox is defined by the top left corner `(x, y)`, the width `w` and
    height `h`. All the values are stored as floats and negative width or height
    are not allowed. The bottom right corner is given by (x + w, y + h).
    """

    # Constructors
    def __init__(self, x: float | int = 0, y: float | int = 0,
                 w: float | int = 1, h: float | int = 1):
        """
        Default constructor for the `BBox` class.
        """
        try:
            self.__x = float(x)
            self.__y = float(y)
            self.__w = float(w)
            self.__h = float(h)
        except Exception as e:
            logger.error(
                f"Error while creating BBox with parameters {x, y, w, h}: {e}."
            )
            raise ValueError(
                f"Error while creating BBox with parameters {x, y, w, h}: {e}."
            )
        self._check()

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> BBox:
        """
        Create a `BBox` from top left x-y and bottom right x-y coordinates.
        """
        return BBox(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def from_two_corners(x1: float, y1: float, x2: float, y2: float) -> BBox:
        """
        Create a `BBox` from any two corners defining the bounding box.
        """
        return BBox.from_xyxy(
            min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        )

    @staticmethod
    def from_points(points: NDArray) -> BBox:
        """
        Create a `BBox` containing all the provided points.

        Inputs
        - points: `NDArray(..., 2)` list of 2D points in format [x, y]
        """
        return BBox.from_xyxy(
            points[..., 0].min(), points[..., 1].min(),
            points[..., 0].max(), points[..., 1].max()
        )

    @staticmethod
    def from_center_wh(x_center: float, y_center: float, w: float, h: float) \
        -> BBox:
        """
        Create a `BBox` from the center x-y coordinates, and width and height.
        """
        return BBox(x_center - 0.5 * w, y_center - 0.5 * h, w, h)

    @staticmethod
    def random(position_max: float = 100,
               size_mean: float = 10, size_std: float = 6) -> BBox:
        """
        Create a random `BBox`.
        """
        w, h = -1, -1
        while w < 0: w = np.random.normal(size_mean, size_std)
        while h < 0: h = np.random.normal(size_mean, size_std)

        x_center = np.random.uniform(0, position_max)
        y_center = np.random.uniform(0, position_max)

        return BBox.from_center_wh(x_center, y_center, w, h)


    def copy(self) -> BBox:
        """
        Return a deep copy of this BBox.
        """
        return BBox(self.x, self.y, self.w, self.h)


    def _check(self):
        """
        Check if the parameters of the BBox are valid or not.
        """
        if self.w < 0 or self.h < 0:
            logger.error(
                "The width and height of a `BBox` can't be negative."
            )
            raise ValueError(
                "The width and height of a `BBox` can't be negative."
            )

    # Properties
    @property
    def x(self) -> float:
       """
       `float`: Left x coordinate. Equivalent to `x1`.
       """
       return self.__x

    @property
    def y(self) -> float:
       """
       `float`: Top y coordinate. Equivalent to `y1`.
       """
       return self.__y

    @property
    def w(self) -> float:
       """
       `float`: Width of the bounding box (>= 0).
       """
       return self.__w

    @property
    def h(self) -> float:
       """
       `float`: Height of the bounding box (>= 0).
       """
       return self.__h

    @property
    def x1(self) -> float:
       """
       `float`: Left x coordinate. Equivalent to `x`.
       """
       return self.__x

    @property
    def y1(self) -> float:
       """
       `float`: Top y coordinate. Equivalent to `y`.
       """
       return self.__y

    @property
    def x2(self) -> float:
        """
        `float`: Right x coordinate. Depends on mode.
        """
        return self.x + self.w

    @property
    def y2(self) -> float:
        """
        `float`: Bottom y coordinate. Depends on mode.
        """
        return self.y + self.h

    @property
    def area(self) -> float:
        """
        `float`: Area of the bounding box.
        """
        return self.w * self.h

    # Equality
    def __eq__(self, x: Any):
        if not isinstance(x, BBox):
            return False

        return list(self) == list(x)

    # Unpack
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h


    # Methods
    def corners(self) -> NDArray:
        """
        Returns the four corner coordinates:
          - in a CCW manner when the x-axis points right and the y-axis down
          - in a CW manner when the x-axis points right and the y-axis up

        Returns
        - corners: `NDArray(4, 2)` array containing the four corners
        """
        return np.array([[self.x1, self.y1],
                         [self.x1, self.y2],
                         [self.x2, self.y2],
                         [self.x2, self.y1]])


    def xywh_tuple(self) -> tuple[float, float, float, float]:
        """
        Returns a `tuple` (x, y, w, h).
        """
        return cast(tuple[float, float, float, float], tuple(self))

    def xywh_array(self) -> NDArray:
        """
        Returns a `NDArray(4,)` [x, y, w, h].
        """
        return np.array([*self.xywh_tuple()])


    def xyxy_tuple(self) -> tuple[float, float, float, float]:
        """
        Returns a `tuple` (x1, y1, x2, y1).
        """
        return self.x1, self.y1, self.x2, self.y2

    def xyxy_array(self) -> NDArray:
        """
        Returns a `NDArray(4,)` [x1, y1, x2, y1].
        """
        return np.array([*self.xyxy_tuple()])

    def xyxy_matrix(self) -> NDArray:
        """
        Returns the top left and bottom right corner coordinates.

        Returns
        - xyxy: `NDArray(2, 2)` array containing the two corners
        """
        return self.xyxy_array().reshape((2,2))


    def center(self) -> NDArray:
        """
        Returns the center coordinates of the bounding box.

        Returns
        - center: `NDArray(2)` array containing the center coordinates
        """
        return np.mean(self.xyxy_matrix(), axis=0)

    def center_wh_tuple(self) -> tuple[float, float, float, float]:
        """
        Returns a `tuple` (x_center, y_center, w, h).
        """
        return *self.center(), self.w, self.h

    def center_wh_array(self) -> NDArray:
        """
        Returns a `NDArray(4,)` [x_center, y_center, w, h].
        """
        return np.array([*self.center_wh_tuple()])


    # Representations
    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"BBox(x={self.x:.1f}, y={self.y:.1f}, " + \
               f"w={self.w:.1f}, h={self.h:.1f})"


    # Intersection
    @staticmethod
    def intersect(bbox_1: BBox, bbox_2: BBox) -> bool:
        """
        Return `True` if the `BBox` intersects with another `BBox`.
        """
        if bbox_1.x2 > bbox_2.x and bbox_1.y2 > bbox_2.y and \
           bbox_1.x < bbox_2.x2 and bbox_1.y < bbox_2.y2:
            return True
        else:
            return False

    def intersect_with(self, bbox: BBox) -> bool:
        """
        Return `True` if the `BBox` intersects with another `BBox`.
        """
        return BBox.intersect(self, bbox)

    @staticmethod
    def intersection(bbox_1: BBox, bbox_2: BBox,
                     intersect_check: Optional[bool] = True) -> BBox:
        """
        Compute the intersection between two `BBox`.

        Inputs
        - bbox_1: `BBox` the first bounding box
        - bbox_2: `BBox` the second bounding box

        Optional inputs
        - intersect_check: `bool` whether to check if the two `BBox` intersect
                           or not. If set to `True` the function will raise an
                           error if the two `BBox` do not intersect. If set to
                           `False`, the function will return a `BBox` even if
                           the two `BBox` do not intersect. Default is `True`.

        Returns
        - intersection: `BBox` the intersection between the two BBoxes
        """
        if intersect_check and not BBox.intersect(bbox_1, bbox_2):
            raise ValueError("The BBoxes do not intersect.")

        return BBox.from_xyxy(
            max(bbox_1.x, bbox_2.x), max(bbox_1.y, bbox_2.y),
            min(bbox_1.x2, bbox_2.x2), min(bbox_1.y2, bbox_2.y2)
        )

    def intersection_with(self, bbox: BBox,
                      intersect_check: Optional[bool] = True) -> BBox:
        """
        Compute the intersection between two `BBox`.

        Inputs
        - bbox: `BBox` the other BBox to intersect with

        Optional inputs
        - intersect_check: `bool` whether to check if the two `BBox` intersect
                           or not. If set to `True` the function will raise an
                           error if the two `BBox` do not intersect. If set to
                           `False`, the function will return a `BBox` even if
                           the two `BBox` do not intersect. Default is `True`.

        Returns
        - intersection: `BBox` the intersection between the two BBoxes
        """
        return BBox.intersection(self, bbox, intersect_check=intersect_check)

    # Operators
    def __contains__(
            self,
            other: tuple[float, float] | list[float] | NDArray | BBox
        ) -> bool:
        # TODO: add support for multiple points, i.e. list of tuples or arrays
        if isinstance(other, tuple):
            if not len(other) == 2:
                logger.error(
                    f"BBox containment check with tuple `{other}` of length " +
                    f"{len(other)} is not supported. Length must be 2."
                )
                raise ValueError(
                    f"BBox containment check with tuple `{other}` of length " +
                    f"{len(other)} is not supported. Length must be 2."
                )
            if not isinstance(other[0], (int, float)) or \
               not isinstance(other[1], (int, float)):
                logger.error(
                    f"BBox containment check with tuple `{other}` is not " +
                    f"supported. Tuple must contain two scalars."
                )
                raise ValueError(
                    f"BBox containment check with tuple `{other}` is not " +
                    f"supported. Tuple must contain two scalars."
                )
            return (self.x <= other[0] <= self.x2 and
                    self.y <= other[1] <= self.y2)
        if isinstance(other, list):
            if not len(other) == 2:
                logger.error(
                    f"BBox containment check with list `{other}` of length " +
                    f"{len(other)} is not supported. Length must be 2."
                )
                raise ValueError(
                    f"BBox containment check with list `{other}` of length " +
                    f"{len(other)} is not supported. Length must be 2."
                )
            if not isinstance(other[0], (int, float)) or \
               not isinstance(other[1], (int, float)):
                logger.error(
                    f"BBox containment check with list `{other}` is not " +
                    f"supported. List must contain two scalars."
                )
                raise ValueError(
                    f"BBox containment check with list `{other}` is not " +
                    f"supported. List must contain two scalars."
                )
            return (other[0], other[1]) in self
        if isinstance(other, np.ndarray):
            if not other.ndim == 1 or not len(other) == 2:
                logger.error(
                    f"BBox containment check with array `{other}` of shape " +
                    f"{other.shape} is not supported. Shape must be (2,)."
                )
                raise ValueError(
                    f"BBox containment check with array `{other}` of shape " +
                    f"{other.shape} is not supported. Shape must be (2,)."
                )
            return (other[0], other[1]) in self
        if isinstance(other, BBox):
            # Check if the other BBox is completely contained in this BBox
            return all(corner in self for corner in other.corners())
        logger.error(
            f"BBox containment check not supported with type `{type(other)}`."
        )
        raise NotImplementedError(
            f"BBox containment check not supported with type `{type(other)}`."
        )


    def __add__(self, d: Any) -> BBox:
        """
        Shifts the bounding BBox by a scalar `d` or by a tuple `(d_x, d_y)`.
        """
        if not isinstance(d, (int, float, tuple)):
            logger.error(
                f"BBox addition with type `{type(d)}` is not supported."
            )
            raise NotImplementedError(
                f"BBox addition with type `{type(d)}` is not supported."
            )
        if isinstance(d, (int, float)):
            return BBox(self.x + d, self.y + d, self.w, self.h)
        if isinstance(d, tuple):
            if len(d) == 2 and all(isinstance(i, (int, float)) for i in d):
                return BBox(self.x + d[0], self.y + d[1], self.w, self.h)
            else:
                logger.error(
                    f"BBox addition with tuple `{d}` is not supported."
                )
                raise NotImplementedError(
                    f"BBox addition with tuple `{d}` is not supported."
                )


    def __mul__(self, scale: Any) -> BBox:
        """
        Scale the bounding box by a scalar.

        Inputs
        - scale: `float` or `int` scale value
        """
        if not isinstance(scale, (int, float)):
            raise NotImplementedError(f"BBox multiplication with type " +
                                      f"{type(scale)} is not supported.")

        return BBox.from_center_wh(*self.center(),
                                   scale * self.w, scale * self.h)


    def __rmul__(self, scale: float) -> BBox:
        """
        Scale the bounding box by a scalar.

        Inputs
        - scale: `float` or `int` scale value
        """
        return self.__mul__(scale)


    # Cython BBox
    @staticmethod
    @requires_package('cython_bbox')
    def iou(b_1: BBox, b_2: BBox) -> float:
        """
        Compute the Intersection-over-Union (IoU) between two BBoxes.

        Inputs
        - b_1: `BBox`
        - b_2: `BBox`

        Returns
        - iou: `float` the IoU between BBox b_1 and BBox b_2
        """
        # Format for cython: xyxy coordinates with `-1` subtracted for the
        # bottom right corner coordinates
        xyxy_1 = (b_1.xyxy_array() + np.array([0, 0, -1, -1]))[None, :]
        xyxy_2 = (b_2.xyxy_array() + np.array([0, 0, -1, -1]))[None, :]

        return cython_bbox.bbox_overlaps(xyxy_1, xyxy_2)[0, 0]


    # Visualization functions
    def show(self, axes: Optional[Axes] = None,
             savefig: Optional[str | Path] = None, **args) -> Axes:
        """
        Visualize the BBox in a matloptlib plot.
        """
        return BBox.visualize(self, axes, savefig, **args)

    @staticmethod
    def visualize(bboxes: BBox | list[BBox],
                  axes: Optional[Axes] = None,
                  savefig: Optional[str | Path] = None,
                  show: Optional[bool] = True,
                  show_text: Optional[bool] = True,
                  color: Optional[NDArray | str] = None,
                  alpha: Optional[float] = None,
                  only_borders: Optional[bool] = False,
                  linewidth: Optional[int] = None,
                  linestyle: Optional[str] = "solid",
                  **args) -> Axes:
        """
        Visualize a BBox or a list of BBoxes in a matloptlib plot.

        Inputs
        - bboxes: list of `BBox` to plot

        Optional Inputs
        - axes:      `Axes` matplotlib axes to plot on. If not provided, a new
                     figure will be created.
        - show:      `bool` whether to show the plot or not. Default is True. If
                     axes are provided, the plot will not be shown. If savefig
                     is provided, the plot will not be shown.
        - savefig:   `str` or `Path` path to save the figure. By default the
                     figure is not being saved.
        - show_text: `bool` whether to show the label of the BBoxes or not.
        - color:     `NDArray(3,)` color of the BBoxes.
        - alpha:     `float` transparency of the BBoxes.
        """

        # Assertions
        if savefig and axes:
            logger.error("Can't save figure if axes are provided.")
            raise ValueError("Can't save figure if axes are provided.")

        # Do not show
        if savefig or axes:
            show = False

        if isinstance(bboxes, BBox):
            bboxes = [bboxes]

        # Save figure
        if savefig:
            axes = None
            show = False

        # No axes provided
        if axes is None:
            # Create figure
            fig = plt.figure()
            ax: Axes = fig.add_subplot()

            # Title
            ax.set_title(f"BBox{'es' if len(bboxes) > 1 else ''} " +
                         f"visualization")

            # Axis labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.invert_yaxis()

        # Axes provided
        else:
            ax = axes

        # Default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [np.array(matplotlib.colors.to_rgb(c)) for c in colors]

        # Color
        if color is not None:
            if isinstance(color, str):
                color = np.array(matplotlib.colors.to_rgb(color))

        # Alpha
        if alpha is None and len(bboxes) > 0:
            alpha = max(0.2, 1/len(bboxes))

        # Plot
        bbox: BBox
        if linewidth is None:
            linewidth = 2 if only_borders else 1
        for i, bbox in enumerate(bboxes):

            c = colors[i%len(colors)] if color is None else color

            facecolor = c if not only_borders else 'none'
            edgecolor = c if only_borders else 0.7 * c

            # Rectangle
            rectangle = matplotlib.patches.Rectangle(
                (bbox.x, bbox.y), bbox.w, bbox.h, alpha=alpha,
                edgecolor=edgecolor, facecolor=facecolor,
                linewidth=linewidth, linestyle=linestyle
                )
            ax.add_patch(rectangle)

            # Labels
            if show_text and (len(bboxes) > 1 or \
                hasattr(bbox, "label") or hasattr(bbox, "name")):
                label = i
                if hasattr(bbox, "name"):
                    label = bbox.name
                elif hasattr(bbox, "label"):
                    label = bbox.label
                ax.text(*bbox.center(), label, ha='center', va='center',
                        alpha=alpha, color=np.array(c) * 0.7,
                        fontsize=12)

        if axes is None:
            # Axis parameters
            ax.axis('equal')

        # Show
        if show: plt.show()

        # Save figure
        if savefig: fig.savefig(savefig)

        return ax
