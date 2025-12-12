# Typing
from typing import Callable, TypeVar, cast, Union

# NumPy
import numpy as np

Point = TypeVar("Point", bound=Union[np.ndarray, float])

def rk4_step(
        f: Callable[[Point, float], Point],
        y0: Point,
        t0: float,
        dt: float
    ) -> Point:
    """
    Runge-Kutta 4th order integration step for the differential equation
        dy(t)/dt = f(y, t).

    Args:
        f (Callable[[float, Point], Point]): The differential function f(t, y)
                                             to integrate.
        y0 (Point): The initial state.
        t0 (float): The initial time.
        dt (float): The time step.

    Returns:
        Point: The state `y` at time `t0 + dt` after integration.
    """

    k1 = f(                          y0,           t0)
    k2 = f(cast(Point, y0 + dt * k1 / 2), t0 + dt / 2)
    k3 = f(cast(Point, y0 + dt * k2 / 2), t0 + dt / 2)
    k4 = f(    cast(Point, y0 + dt * k3),     t0 + dt)

    y = cast(Point, y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return y
