# Pytest
import pytest

# Numpy
import numpy as np

# Utils
from alutils import Color

def test_color_constructor():

    assert Color() == Color("red") == Color(255, 0, 0)

    c_blue = Color("blue")
    c_blue_transparent = c_blue.with_opacity(0.8)
    assert not c_blue == c_blue_transparent
    assert not c_blue == c_blue.with_opacity(1.)

    assert c_blue == Color("blue")
    assert c_blue == Color(0, 0, 255)
    assert c_blue == Color(r=0, g=0, b=255)
    assert c_blue == Color(red=0, green=0, blue=255)
    assert c_blue == Color(b=255, r=0, g=0)
    assert c_blue == Color(0., 0., 1.)
    assert c_blue == Color((0, 0, 255))
    assert c_blue == Color((0., 0., 1.))
    assert c_blue == Color([0, 0, 255])
    assert c_blue == Color([0., 0., 1.])
    assert c_blue == Color(np.array([0, 0, 255]))
    assert c_blue == Color(np.array([0., 0., 1.]))
    assert c_blue == Color("rgb(0, 0, 255)")

    assert c_blue_transparent == Color("blue", opacity=0.8)
    assert c_blue_transparent == Color(0, 0, 255, 0.8)
    assert c_blue_transparent == Color(r=0, g=0, b=255, a=0.8)
    assert c_blue_transparent == Color(0., 0., 1., 0.8)
    assert c_blue_transparent == Color((0, 0, 255), opacity=0.8)
    assert c_blue_transparent == Color((0., 0., 1.), opacity=0.8)
    assert c_blue_transparent == Color([0, 0, 255], opacity=0.8)
    assert c_blue_transparent == Color([0., 0., 1.], opacity=0.8)
    assert c_blue_transparent == Color(np.array([0, 0, 255]), opacity=0.8)
    assert c_blue_transparent == Color(np.array([0., 0., 1.]), opacity=0.8)
    assert c_blue_transparent == Color(np.array([0, 0, 255]), alpha=0.8)
    assert c_blue_transparent == Color(np.array([0., 0., 1.]), alpha=0.8)
    assert c_blue_transparent == Color(np.array([0, 0, 255]), a=0.8)
    assert c_blue_transparent == Color(np.array([0., 0., 1.]), a=0.8)
    assert c_blue_transparent == Color("rgb(0, 0, 255)", opacity=0.8)

    with pytest.raises(Exception): Color(0, 0, 255, 2)
    with pytest.raises(Exception): Color(r=0, g=0, b=255, a=2)
    with pytest.raises(Exception): Color(red=3, green=10, blue=255, a=2)
    with pytest.raises(Exception): Color(red=3, green=10, blue=255, opacity=2)
    with pytest.raises(Exception): Color(0., 0, 1)
    with pytest.raises(Exception): Color((0., 0, 1))
    with pytest.raises(Exception): Color([0., 0, 1])
    with pytest.raises(Exception): Color(np.array([0, 0, 1.2]))
    with pytest.raises(Exception): Color(np.array([0, 123, 1.5]))
    with pytest.raises(Exception): Color(0., -0.2, 0.5)
    with pytest.raises(Exception): Color("rgb(0, 12, 323)")
    with pytest.raises(Exception): Color("rgb(0, 0.1, 1)")
    with pytest.raises(Exception): Color("rgb(0.2, 0.1, 1.0)")
    with pytest.raises(Exception): Color("rgba(0, 12, 323)")
    with pytest.raises(Exception): Color("rgba(0, 0, 123, 234)")
    with pytest.raises(Exception): Color("rgba(0, 0, 123, -0.1)")
    with pytest.raises(Exception): Color("rgba(0, 0, 123, 1.1)")
    with pytest.raises(Exception): Color("rgba(0.9, 0, 123, 0.5)")
    with pytest.raises(Exception): Color("rgba(0.9, 0.2, -0.8, 0.5)")

    with pytest.raises(Exception):
        Color("rgba(100, 100, 100, 0.5)", opacity=0.5)
    with pytest.raises(Exception):
        Color("rgba(100, 100, 100, 0.5)", alpha=0.5)

    assert not Color("lime") == Color("lime").with_opacity(1.)
    with pytest.raises(Exception): Color("lime", opacity=1)
    with pytest.raises(Exception): Color("lime", opacity=1.2)
    with pytest.raises(Exception): Color("lime", opacity=0)
    with pytest.raises(Exception): Color("lime", opacity=255)

    with pytest.raises(Exception): Color("lime", 0.5)
    with pytest.raises(Exception): Color("lime", opacity=0.5, alpha=0.5)
    with pytest.raises(Exception): Color("lime", opacity=0.5, a=0.5)
    with pytest.raises(Exception): Color("lime", alpha=0.5, a=0.5)

    with pytest.raises(Exception):
      Color("#bd88ab67", opacity=0.5)
    Color("#b81b84")
    Color("#b81b84", opacity=0.5)
    Color("#b81b8458")

    c = Color("#88fb14")
    assert c.hex == "#88fb14"

    c = Color("#88fb14", opacity=0.5)
    assert c.hex == "#88fb1480"

    c = Color("#b81b8443")
    assert c.hex == "#b81b8443"

