from mesh import SingleBlockMesh, SingleBlockExtrudedMesh
import numpy as np
from ufl import SpatialCoordinate, as_vector
from function import Function
from constant import Constant


def create_box_mesh(nxs, lxs, pxs, bcs):
    assert(len(nxs) == len(lxs))
    assert(len(nxs) == len(pxs))
    assert(len(nxs) == len(bcs))

    pxs = np.array(pxs, dtype=np.float32)
    lxs = np.array(lxs, dtype=np.float32)
    nxs = np.array(nxs, dtype=np.int32)

    mesh = SingleBlockMesh(nxs, bcs)

    xs = SpatialCoordinate(mesh)
    newcoords = Function(mesh.coordinates.function_space(), name='newcoords')
    xlist = []
    for i in range(len(pxs)):
        xlist.append(xs[i] * Constant(lxs[i]) + Constant(pxs[i]) - Constant(lxs[i])/2.)
    newcoords.interpolate(as_vector(xlist))
    mesh.coordinates.assign(newcoords)

    return mesh

def Mesh(newcoords):
    oldmesh = newcoords.space._mesh
    if not oldmesh.extruded:
        return SingleBlockMesh(oldmesh.nxs, oldmesh.bcs, coords=newcoords)
    if oldmesh.extruded:
        basemesh = SingleBlockMesh(oldmesh.nxs[:-1], oldmesh.bcs[:-1])
        return SingleBlockExtrudedMesh(basemesh, oldmesh.nxs[-1], coords=newcoords)

def ExtrudedMesh(basemesh, layers, layer_height=None, extrusion_type='uniform'):
    if not (extrusion_type == 'uniform'):
        raise ValueError('cannot handle non-uniform extrusion yet')

    emesh = SingleBlockExtrudedMesh(basemesh, layers, layer_height=layer_height)

    return emesh



def PeriodicIntervalMesh(ncells, length_or_left, right=None):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if ``right``
             is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
             boundary point (in which case ``length_or_left`` should
             be the left boundary point).

    Creates a periodic domain with extent [0,length] OR extent [left,right]
    """

    if right is None:
        left = 0
        right = length_or_left
    else:
        left = length_or_left

    if ncells <= 0 or ncells % 1:
        raise ValueError("Number of cells must be a postive integer")
    length = right - left
    if length < 0:
        raise ValueError("Requested mesh has negative length")

    nxs = [ncells, ]
    lxs = [length, ]
    pxs = [length/2.0, ]
    bcs = ['periodic', ]
    return create_box_mesh(nxs, lxs, pxs, bcs)


def PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="both", quadrilateral=True):
    """Generate a periodic rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg direction: The direction of the periodicity, one of
            ``"both"``, ``"x"`` or ``"y"``.

    Creates a domain with extent [0,Lx] x [0,Ly]

    """
    if not quadrilateral:
        raise ValueError("Themis does not support non-quadrilateral cells in 2D")
    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    nxs = [nx, ny]
    lxs = [Lx, Ly]
    pxs = [Lx/2.0, Ly/2.0]
    if direction == 'both':
        bcs = ['periodic', 'periodic']
    if direction == 'x':
        bcs = ['periodic', 'nonperiodic']
    if direction == 'y':
        bcs = ['nonperiodic', 'periodic']
    return create_box_mesh(nxs, lxs, pxs, bcs)


def PeriodicSquareMesh(nx, ny, L, direction="both", quadrilateral=True):
    """Generate a periodic square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :arg direction: The direction of the periodicity, one of
            ``"both"``, ``"x"`` or ``"y"``.

    Creates a domain with extent [0,L] x [0,L]

    """
    return PeriodicRectangleMesh(nx, ny, L, L, direction=direction, quadrilateral=quadrilateral)


def IntervalMesh(ncells, length_or_left, right=None):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if ``right``
     is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
     boundary point (in which case ``length_or_left`` should
     be the left boundary point).

    Creates a domain with extent [0,length] OR extent [left,right]
    """
    if right is None:
        left = 0
        right = length_or_left
    else:
        left = length_or_left

    if ncells <= 0 or ncells % 1:
        raise ValueError("Number of cells must be a postive integer")
    length = right - left
    if length < 0:
        raise ValueError("Requested mesh has negative length")

    nxs = [ncells, ]
    lxs = [length, ]
    pxs = [length/2.0, ]
    bcs = ['nonperiodic', ]
    return create_box_mesh(nxs, lxs, pxs, bcs)


def SquareMesh(nx, ny, L, quadrilateral=True):
    """Generate a square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions

    Creates a domain with extent [0,L] x [0,L]
    """
    return RectangleMesh(nx, ny, L, L, quadrilateral=quadrilateral)


def CubeMesh(nx, ny, nz, L, xbc, ybc, zbc):
    """Generate a cube mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg L: The extent in the x, y and z directions
    :arg xbc: The periodicity in the x direction, either periodic or nonperiodic
    :arg ybc: The periodicity in the y direction, either periodic or nonperiodic
    :arg zbc: The periodicity in the z direction, either periodic or nonperiodic

    Creates a domain with extent [0,L] x [0,L] x [0,L]
    """
    return BoxMesh(nx, ny, nz, L, L, L, xbc, ybc, zbc)


def BoxMesh(nx, ny, nz, Lx, Ly, Lz, xbc, ybc, zbc):
    """Generate a box mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :arg xbc: The periodicity in the x direction, either periodic or nonperiodic
    :arg ybc: The periodicity in the y direction, either periodic or nonperiodic
    :arg zbc: The periodicity in the z direction, either periodic or nonperiodic

    Creates a domain with extent [0,Lx] x [0,Ly] x [0,Lz]
    """

    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    nxs = [nx, ny, nz]
    lxs = [Lx, Ly, Lz]
    pxs = [Lx/2.0, Ly/2.0, Lz/2.0]
    bcs = [xbc, ybc, zbc]
    return create_box_mesh(nxs, lxs, pxs, bcs)


def RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True):
    """Generate a rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction

    Creates a domain with extent [0,Lx] x [0,Ly]
    """

    if not quadrilateral:
        raise ValueError("Themis does not support non-quadrilateral cells in 2D")
    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    nxs = [nx, ny]
    lxs = [Lx, Ly]
    pxs = [Lx/2.0, Ly/2.0]
    bcs = ['nonperiodic', 'nonperiodic']
    return create_box_mesh(nxs, lxs, pxs, bcs)
