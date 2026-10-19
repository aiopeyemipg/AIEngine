"""
Stand-in for the `long_side_form` module imported by ../ABC/*-ABC.py,
which is not part of that folder: puts an ellipse in long-side form
(a >= b, swapping the axes turns theta by pi/2; theta wrapped to [0, pi)).
Used only by scripts/run_python_reference.py.
"""
import math


def to_long_side_form(cx, cy, a, b, theta):
    if b > a:
        a, b = b, a
        theta += math.pi / 2.0
    theta = theta % math.pi
    return cx, cy, a, b, theta
