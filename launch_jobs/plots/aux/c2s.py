"""
    cast coordinates between cartesian and spherical
"""
from typing import Tuple, Union
from math import sin, cos, atan2, sqrt

Number = Union[int, float]
Vector = Tuple[Number, Number, Number]


def distance(a: Vector, b: Vector) -> Number:
    """Returns the distance between two cartesian points."""
    x = (b[0] - a[0]) ** 2
    y = (b[1] - a[1]) ** 2
    z = (b[2] - a[2]) ** 2
    return (x + y + z) ** 0.5

  
def magnitude(x: Number, y: Number, z: Number) -> Number:
    """Returns the magnitude of the vector."""
    return sqrt(x * x + y * y + z * z)


def to_spherical(x: Number, y: Number, z: Number) -> Vector:
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
    radius = magnitude(x, y, z)
    theta = atan2(sqrt(x * x + y * y), z)
    phi = atan2(y, x)
    return (radius, theta, phi)


def to_cartesian(radius: Number, theta: Number, phi: Number) -> Vector:
    """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
    x = radius * cos(phi) * sin(theta)
    y = radius * sin(phi) * sin(theta)
    z = radius * cos(theta)
    return (x, y, z)