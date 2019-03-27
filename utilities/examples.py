
from interop import SpatialCoordinate, Function, as_vector, sin, pi

__all__ = ['adjust_coordinates', ]


def adjust_coordinates(mesh, c):
    # Distort coordinates
    xs = SpatialCoordinate(mesh)
    newcoords = Function(mesh.coordinates.function_space(), name='newcoords')
    if len(xs) == 1:
        xlist = [xs[0] + c * sin(2*pi*xs[0]), ]
    if len(xs) == 2:
        xlist = [xs[0] + c * sin(2*pi*xs[0])*sin(2*pi*xs[1]), xs[1] + c * sin(2*pi*xs[0])*sin(2*pi*xs[1])]
    if len(xs) == 3:
        xlist = [xs[0] + c * sin(2*pi*xs[0])*sin(2*pi*xs[1])*sin(2*pi*xs[2]), xs[1] + c * sin(2*pi*xs[0])*sin(2*pi*xs[1])*sin(2*pi*xs[2]), xs[2] + c * sin(2*pi*xs[0])*sin(2*pi*xs[1])*sin(2*pi*xs[2])]
    newcoords.interpolate(as_vector(xlist))
    mesh.coordinates.assign(newcoords)
