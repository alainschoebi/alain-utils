from __future__ import annotations

# Typing
from typing import cast, Iterator, ClassVar

# Python
import re

# NumPy
import numpy as np
from numpy.typing import NDArray

# Matplotlib
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

# Manim
try:
    from manim import ManimColor
except:
    pass

# Utils
from alutils.decorators import requires_package
from alutils.types import Number

class Color:
    """
    Color class to represent colors in RGB or RGBA format.

    All colors are internally represented as `float` values in `[0., 1.]`.

    The `opacity` or `alpha` value is optional and can be set to `None` if not
    needed.
    """

    # Predefine colors
    BLACK: ClassVar["Color"]
    WHITE: ClassVar["Color"]
    RED: ClassVar["Color"]
    GREEN: ClassVar["Color"]
    BLUE: ClassVar["Color"]

    def __init__(self, *args, **kwargs):
        """
        General color constructor.

        The `opacity`, `alpha`, or `a` arguments must always be provided as
        `float` in `[0., 1.]`.

        Available constructors:
        - Default color (red).
            Color()
        - RGB color with `int` values in `[0, 255]`:
            Color(`int`, `int`, `int`)
        - RGB color with `float` values in `[0., 1.]`:
            Color(`float`, `float`, `float`)
        - RGB color with `tuple` values in `[0, 255]`:
            Color((`int`, `int`, `int`))
        - RGB color with `tuple` values in `[0., 1.]`:
            Color((`float`, `float`, `float`))
        - RGB color with `list` values in `[0, 255]`:
            Color([`int`, `int`, `int`])
        - RGB color with `list` values in `[0., 1.]`:
            Color([`float`, `float`, `float`])
        - RGB color with `NDArray` values in `[0, 255]`:
            Color(np.array([`int`, `int`, `int`]))
        - RGB color with `NDArray` values in `[0., 1.]`:
            Color(np.array([`float`, `float`, `float`]))
        - RGB color with string `rgb(r, g, b)` format:
            Color("rgb(`int`, `int`, `int`)")
        - RGB color with string name:
            Color("<color name>")
        - RGB color from HSV values:
            Color(h=`float`, s=`float`, v=`float`)
        - RGBA colors ...

        Note: the keyword arguments `opacity`, `alpha` and `a` are aliases for
              the opacity value. Only one of them should be provided.

        Non-exhaustive list of example types:
        - Color()
        - Color(opacity=0.5)
        - Color("red")
        - Color("red", opacity=0.5)
        - Color(234, 12, 120)
        - Color(234, 12, 120, 0.5)
        - Color(234, 12, 120, opacity=0.5)
        - Color(0.8, 0.1, 0.8)
        - Color(0.8, 0.1, 0.8, 0.5)
        - Color(0.8, 0.1, 0.8, opacity=0.5)
        - Color((234, 12, 120))
        - Color((234, 12, 120, 0.5))
        - Color((234, 12, 120), opacity=0.5)
        - Color((0.8, 0.1, 0.8))
        - Color((0.8, 0.1, 0.8, 0.5))
        - Color((0.8, 0.1, 0.8), opacity=0.5)
        - Color([234, 12, 120])
        - Color([234, 12, 120, 0.5])
        - Color([234, 12, 120], opacity=0.5)
        - Color([0.8, 0.1, 0.8])
        - Color([0.8, 0.1, 0.8, 0.5])
        - Color([0.8, 0.1, 0.8], opacity=0.5)
        - Color(np.array([234, 12, 120]))
        - Color(np.array([234, 12, 120]), opacity=0.5)
        - Color(np.array([0.8, 0.1, 0.8]))
        - Color(np.array([0.8, 0.1, 0.8, 0.5]))
        - Color(np.array([0.8, 0.1, 0.8]), opacity=0.5)
        - Color("rgb(255, 12, 210)")
        - Color("rgba(255, 255, 231, 0.5)")
        - Color("rgb(255, 12, 120)", opacity=0.5)
        - Color(r=255, g=12, b=120)
        - Color(r=255, g=12, b=120, a=0.5)
        - Color(r=255, g=12, b=120, opacity=0.5)
        - Color(red=255, green=12, blue=120)
        - Color(red=255, green=12, blue=120, alpha=0.5)
        - Color(red=255, green=12, blue=120, opacity=0.5)
        - Color(h=0.8, s=0.1, v=0.8)
        - Color(h=0.8, s=0.1, v=0.8, opacity=0.5)
        - Color(hue=0.8, saturation=0.1, value=0.8)
        - Color(hue=0.8, saturation=0.1, value=0.8, opacity=0.5)
        """

        self.__a = None

        # Verify opacity input
        if sum([x in kwargs for x in ["opacity", "alpha", "a"]]) > 1:
            raise ValueError(
                "The kwargs `opacity`, `alpha` and `a` are aliases. " +
                "Please provide only one of them."
            )

        # Remove opacity, alpha or a from kwargs
        # Check if opacity is provided and extract it
        kwargs_original = kwargs.copy()
        opacity_provided = "opacity" in kwargs or \
                           "alpha" in kwargs or \
                           "a" in kwargs
        opacity = kwargs.pop("opacity", None)
        opacity = kwargs.pop("alpha", None) if opacity is None else opacity
        opacity = kwargs.pop("a", None) if opacity is None else opacity

        # Positional arguments
        if len(args) == 1:
            arg = args[0]

            # tuple
            if isinstance(arg, tuple):
                if len(arg) not in (3, 4):
                    raise ValueError(
                        f"Invalid tuple length for color constructor: " + \
                        f"{len(arg)}. Expected 3 or 4."
                    )
                if not type(arg[0]) == type(arg[1]) == type(arg[2]) or \
                   not isinstance(arg[0], (int, float)):
                    raise ValueError(
                        f"The r, g, b arguments must be of the same type " + \
                        f"and either of type `int` or `float`. Got " + \
                        f"{type(arg[0])}, {type(arg[1])} and {type(arg[2])}."
                    )

                self.r, self.g, self.b = arg[:3]
                if len(arg) == 4:
                    self.a = arg[3]

            # list
            elif isinstance(arg, list):
                Color.__init__(self, tuple(arg))

            # NDArray
            elif isinstance(arg, np.ndarray):
                if arg.ndim != 1 or len(arg) not in (3, 4):
                    raise ValueError(
                        f"Invalid NDArray length for color constructor: " + \
                        f"{len(arg)}. Expected 3 or 4."
                    )
                if not arg.dtype in (
                    np.int32, np.int64, np.float32, np.float64
                   ):
                    raise ValueError(
                        f"Invalid NDArray dtype for color constructor: " + \
                        f"{arg.dtype}. Expected `int` or `float`."
                    )
                Color.__init__(self, arg.tolist())

            # String
            elif isinstance(arg, str):
                # Examples: "rgb(255, 12, 120)" or "rgba(255, 255, 231, 135)"
                if arg.startswith("rgb"):

                    def invalid_string():
                        raise ValueError(
                            f"Invalid string format for color constructor: " + \
                            f'{arg}. Expected "rgb(r, g, b)" or ' + \
                            f'rgba(r, g, b, a)", with r, g, b in `int` in ' + \
                            f"[0, 255] and a in `float` in `[0., 1.]`."
                        )

                    try:
                        values = tuple(map(float, re.findall(r'\d+', arg)))
                    except:
                        invalid_string()
                        raise

                    if not len(values) == 3 or len(values) == 4 or \
                       not all(int(v) == float(v) for v in values[:3]):
                        invalid_string()

                    a = values[3] if len(values) == 4 else None
                    Color.__init__(self, tuple(int(v) for v in values[:3]), a=a)

                # Hexadecimal color
                elif arg.startswith("#"):
                    try:
                        if len(arg) == 7:
                            Color.__init__(self, mcolors.to_rgb(arg))
                        elif len(arg) == 9:
                            Color.__init__(self, mcolors.to_rgba(arg))
                        else:
                            raise ValueError(
                                f"Invalid hexadecimal color string length: " +
                                f"{len(arg)}. Expected 7 or 9 characters."
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Invalid string for hexadecimal color " +
                            f"constructor. Error: {arg}. "
                        )

                # Known strings
                else:
                    try:
                        rgb = mcolors.to_rgb(arg)
                        self.__r, self.__g, self.__b, = rgb[:3]
                    except Exception:
                        raise ValueError(
                            f"Invalid string for color constructor: {arg}."
                        )

            # None of the above types
            else:
                raise TypeError(
                    f"Invalid first positional argument type for color " + \
                    f"construcdtor: {type(arg)}. Expected `str`, `tuple`, " + \
                    f"`list` or `NDArray`."
                )

        # Three or four positional arguments
        elif len(args) == 3 or len(args) == 4:
            Color.__init__(self, tuple(args))

        # Invalid number of positional arguments
        elif not len(args) == 0:
            raise ValueError(
                f"Invalid number of positional arguments (`*args`). " +
                f"Expected 0, 1, 3 or 4 arguments, not {len(args)}."
            )

        # Keyword arguments
        # Default color if no arguments provided (red color)
        elif len(args) == 0 and len(kwargs) == 0:
            Color.__init__(self, "red")

        # hsv argument
        elif len(args) == 0 and len(kwargs) == 1:
            if "hsv" in kwargs:
                hsv = kwargs["hsv"]
                if not isinstance(hsv, (tuple, list)) or \
                   not len(hsv) == 3:
                    raise ValueError(
                        f"Invalid hsv argument for color constructor: " + \
                        f"{hsv}. Expected a `tuple` or `list` of length 3."
                    )
                self.__r, self.__g, self.__b = Color.from_hsv(*hsv)

        # r, g, b and h, s, v keyword arguments
        elif len(args) == 0 and len(kwargs) == 3:
            if "r" in kwargs and "g" in kwargs and "b" in kwargs:
                Color.__init__(self, kwargs["r"], kwargs["g"], kwargs["b"])

            elif "red" in kwargs and "green" in kwargs and "blue" in kwargs:
                Color.__init__(
                    self, kwargs["red"], kwargs["green"], kwargs["blue"]
                )

            elif "h" in kwargs and "s" in kwargs and "v" in kwargs:
                c = Color.from_hsv(kwargs["h"], kwargs["s"], kwargs["v"])
                self.__r, self.__g, self.__b = c

            elif "hue" in kwargs and "saturation" in kwargs and \
                 "value" in kwargs:
                c = Color.from_hsv(
                    kwargs["hue"], kwargs["saturation"], kwargs["value"]
                )
                self.__r, self.__g, self.__b = c

        else:
            raise ValueError(
                f"Not recognized color constructor input: {args} " +
                f"and {kwargs_original}."
            )

        # Opacity or alpha provided
        if opacity_provided:
            if self.has_alpha:
                raise ValueError(
                    "Opacity was provided, but the color already has an " + \
                    "alpha value set."
                )
            self.a = opacity

    @staticmethod
    def from_hsv(h: float, s: float, v: float,
                 opacity: float | None = None) -> Color:
        """
        Create a color from HSV values.

        Inputs:
        - h: `float` hue value in `[0., 1.]`
        - s: `float` saturation value in `[0., 1.]`
        - v: `float` value (brightness) in `[0., 1.]`
        """
        hsv = (h, s, v)
        if not all(isinstance(x, float) for x in hsv) or \
           not all(0 <= x <= 1 for x in hsv):
            raise TypeError("The HSV values must be `float` in `[0., 1.]`.")

        return Color(mcolors.hsv_to_rgb(hsv), opacity=opacity)

    @staticmethod
    def random() -> Color:
        return Color(np.random.rand(3))

    # R, G, B, A properties
    @property
    def r(self) -> float:
        """ Red value `float`"""
        return self.__r

    @property
    def r_int(self) -> int:
        """ Red value `int` """
        return int(self.r * 255)

    @property
    def g(self) -> float:
        """ Green value `float` """
        return self.__g

    @property
    def g_int(self) -> int:
        """ Green value `int` """
        return int(self.g * 255)

    @property
    def b(self) -> float:
        """ Blue value `float` """
        return self.__b

    @property
    def b_int(self) -> int:
        """ Blue value `int` """
        return int(self.b * 255)

    @property
    def a(self) -> float | None:
        """ Alpha value `float` or `None` """
        return self.__a

    @property
    def has_alpha(self) -> bool:
        """ Check if alpha value is set """
        return self.a is not None

    # R, G, B, A setters
    @r.setter
    def r(self, r: float | int) -> None:
        """ Set red value """
        if isinstance(r, float):
            if not 0 <= r <= 1:
                raise ValueError(f"Invalid red value: {r}.")
            self.__r = r
        elif isinstance(r, int):
            if not 0 <= r <= 255:
                raise ValueError(f"Invalid red value: {r}.")
            self.__r = r / 255
        else:
            raise TypeError("Red value must be a `float` or an `int`.")

    @g.setter
    def g(self, g: float | int) -> None:
        """ Set green value with `float` or `int` """
        if isinstance(g, float):
            if not 0 <= g <= 1:
                raise ValueError(f"Invalid green value: {g}.")
            self.__g = g
        elif isinstance(g, int):
            if not 0 <= g <= 255:
                raise ValueError(f"Invalid green value: {g}.")
            self.__g = g / 255
        else:
            raise TypeError("Green value must be a `float` or an `int`.")

    @b.setter
    def b(self, b: float | int) -> None:
        """ Set blue value with `float` or `int` """
        if isinstance(b, float):
            if not 0 <= b <= 1:
                raise ValueError(f"Invalid blue value: {b}.")
            self.__b = b
        elif isinstance(b, int):
            if not 0 <= b <= 255:
                raise ValueError(f"Invalid blue value: {b}.")
            self.__b = b / 255
        else:
            raise TypeError("Blue value must be a `float` or an `int`.")

    @a.setter
    def a(self, a: float | None) -> None:
        """ Set alpha value with `float` or `None` """
        if a is None:
            self.__a = None
        elif isinstance(a, float):
            if not 0 <= a <= 1:
                raise ValueError(f"Invalid alpha value: {a}.")
            self.__a = a
        else:
            raise TypeError("Alpha value must be a `float` or `None`.")

    # Unpacking
    def __iter__(self: Color) -> Iterator[float]:
        """ Unpack the color to `(r, g, b)` or `(r, g, b, a)` """
        if self.has_alpha:
            return iter(self.rgba_tuple())
        else:
            return iter(self.rgb_tuple)

    # RGB and RGBA properties
    @property
    def rgb_tuple(self) -> tuple[float, float, float]:
        """ RGB tuple `(float, float, float)` """
        return (self.r, self.g, self.b)

    @property
    def rgb_int_tuple(self) -> tuple[int, int, int]:
        """ RGB tuple `(int, int, int)` """
        return (self.r_int, self.g_int, self.b_int)

    @property
    def rgb_list(self) -> list[float]:
        """ RGB list `[float, float, float]` """
        return list(self.rgb_tuple)

    @property
    def rgb_int_list(self) -> list[int]:
        """ RGB list `[int, int, int]` """
        return list(self.rgb_int_tuple)

    @property
    def rgb_array(self) -> NDArray:
        """ RGB array `NDArray` """
        return np.array(self.rgb_tuple)

    @property
    def rgb_int_array(self) -> NDArray:
        """ RGB array `NDArray` """
        return np.array(self.rgb_int_tuple)

    def rgba_tuple(self, opacity: float | None = None) \
        -> tuple[float, float, float, float]:
        """ RGBA tuple `(float, float, float, float)` """
        color = self.with_opacity(opacity)
        return color.rgb_tuple + (cast(float, color.a),)

    def rgba_int_tuple(self, opacity: float | None = None) \
        -> tuple[int, int, int, float]:
        """ RGBA tuple `(int, int, int, float)` """
        color = self.with_opacity(opacity)
        return color.rgb_int_tuple + (cast(float, color.a),)

    def rgba_list(self, opacity: float | None = None) \
        -> list[float]:
        """ RGBA list `[float, float, float, float]` """
        color = self.with_opacity(opacity)
        return list(color.rgba_tuple())

    def rgba_int_list(self, opacity: float | None = None) \
        -> list[int | float]:
        """ RGBA list `[int, int, int, float]` """
        color = self.with_opacity(opacity)
        return list(color.rgba_int_tuple())

    def rgba_array(self, opacity: float | None = None) -> NDArray:
        """ RGBA array `NDArray` """
        color = self.with_opacity(opacity)
        return np.array(color.rgba_tuple())

    # Strings
    def rgb_string(self) -> str:
        """ RGB string `rgb(int, int, int)` """
        return f"rgb({self.r_int}, {self.g_int}, {self.b_int})"

    def rgba_string(self, opacity: float | None = None) -> str:
        """ RGBA string `rgba(int, int, int, float)` """
        color = self.with_opacity(opacity)
        return f"rgba({color.r_int}, {color.g_int}, {color.b_int}, {color.a})"

    # Hexadecimal string
    @property
    def hex(self) -> str:
        """ Hexadecimal string `#rrggbb` or `#rrggbbaa` """
        if self.has_alpha:
            return mcolors.to_hex(self.rgba_tuple(), keep_alpha=True)
        else:
            return mcolors.to_hex(self.rgb_tuple)

    # Manim
    @requires_package("manim")
    def to_manim(self) -> ManimColor:
        return ManimColor(self.rgb_tuple)

    # HSV representation
    @property
    def hsv_tuple(self) -> tuple[float, float, float]:
        """ HSV tuple `(float, float, float)` """
        hsv = mcolors.rgb_to_hsv(self.rgb_tuple) # Numpy (3,)
        return (float(hsv[0]), float(hsv[1]), float(hsv[2]))

    @property
    def hsv_list(self) -> list[float]:
        """ HSV list `[float, float, float]` """
        return list(self.hsv_tuple)

    @property
    def hsv_array(self) -> NDArray:
        """ HSV array `NDArray` """
        return np.array(self.hsv_tuple)

    @property
    def h(self) -> float:
        """ Hue value `float` """
        return self.hsv_tuple[0]

    @property
    def s(self) -> float:
        """ Saturation value `float` """
        return self.hsv_tuple[1]

    @property
    def v(self) -> float:
        """ Value (brightness) `float` """
        return self.hsv_tuple[2]

    @property
    def hue(self) -> float:
        """ Hue value `float` """
        return self.hsv_tuple[0]

    @property
    def saturation(self) -> float:
        """ Saturation value `float` """
        return self.hsv_tuple[1]

    @property
    def value(self) -> float:
        """ Value (brightness) `float` """
        return self.hsv_tuple[2]

    # Operations
    def __mul__(
            self: Color, scalar: Number
        ) -> Color:
        """ Multiply color rgb(a) components by a scalar """
        if not isinstance(scalar, Number):
            raise TypeError("Invalid type for multiplication.")
        if scalar < 0:
            raise ValueError("Scalar must be a non-negative number.")

        if self.has_alpha:
            return Color(np.clip(self.rgba_array() * scalar, 0, 1))
        else:
            return Color(np.clip(self.rgb_array * scalar, 0, 1))

    def __rmul__(
            self: Color, scalar: Number
        ) -> Color:
        """ Multiply color rgb(a) components by a scalar """
        return self * scalar

    def __truediv__(
            self: Color, scalar: Number
        ) -> Color:
        """ Divide color rgb(a) components by a scalar """
        return self * (1/scalar)

    # Equality
    def __eq__(self: Color, other: object) -> bool:
        """ Check if two colors are equal """
        if not isinstance(other, Color):
            return False

        if not self.has_alpha == other.has_alpha:
            return False

        return tuple(self) == tuple(other)

    # Interpolation
    @staticmethod
    def interpolate(
            color1: Color, color2: Color, t: float = 0.5
        ) -> Color:
        """ Interpolate between two colors """
        if not 0 <= t <= 1:
            raise ValueError(
                "Interpolation factor `t` must be between 0 and 1."
            )

        if not color1.has_alpha == color2.has_alpha:
            raise ValueError(
                "The two colors must either both have an alpha value or not."
            )

        if color1.has_alpha:
            return Color(
                color1.rgba_array() + \
                    (color2.rgba_array() - color1.rgba_array()) * t
            )
        else:
            return Color(
                color1.rgb_array + (color2.rgb_array - color1.rgb_array) * t
            )

    # With opacity
    def with_opacity(self: Color, opacity: float | None = None) -> Color:
        """
        Return a new `Color` with the same RGB values and the provided opacity.

        Note: If the color already has an alpha value, it will be replaced by
              the provided opacity.
        Note: If the color does not have an alpha value and no opacity is
              provided, a ValueError is raised.
        """
        if not self.has_alpha and opacity is None:
            raise ValueError(
                "Alpha value is `None` and no `opacity` is provided."
            )

        if opacity is None:
            return self

        if not isinstance(opacity, float) or not 0 <= opacity <= 1:
            raise ValueError(
                "Opacity must be a `float` between 0 and 1."
            )

        return Color(self.rgb_tuple + (opacity,))

    # Color transformations
    def complementary_color(self: Color) -> Color:
        v = 2 if self.v < 0.5 else 0.5
        return Color(hsv=(self.h, self.s, v), opacity=self.a)

    def brighter(self: Color, factor: float = 0.5) -> Color:
        """ Return a brighter `Color`. """
        if factor < 0 or factor > 1:
            raise ValueError("Factor must be between 0 and 1.")
        return Color.interpolate(self, Color("white"), factor)
        #v = max(0., min(1., self.v * factor))
        #return Color(hsv=(self.h, self.s, v), opacity=self.a)

    def darker(self: Color, factor: float = 0.5) -> Color:
        """ Return a darker `Color`. """
        if factor < 0 or factor > 1:
            raise ValueError("Factor must be between 0 and 1.")
        return Color.interpolate(self, Color("black"), factor)

    # String representation
    def __str__(self: Color) -> str:
        """ String representation of the color """
        if self.has_alpha:
            return f"Color({self.rgba_string()})"
        else:
            return f"Color({self.rgb_string()})"

    def __repr__(self: Color) -> str:
        """ String representation of the color """
        return self.__str__()

    def show(self: Color) -> None:
        """ Show the color """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow([[self.rgb_array]], aspect='auto')
        ax.set_title(str(self))
        ax.axis('off')
        plt.show()

Color.BLACK = Color("black")
Color.WHITE = Color("white")
Color.RED = Color("red")
Color.GREEN = Color("green")
Color.BLUE = Color("blue")
