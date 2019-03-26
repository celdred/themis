
backend = 'themis'

import matplotlib.pyplot as plt
import numpy as np

if backend == 'firedrake':
    from firedrake import IntervalMesh, PeriodicIntervalMesh, PeriodicSquareMesh, SquareMesh, ExtrudedMesh
    from firedrake import DirichletBC
    from firedrake import Constant
    from firedrake import as_vector, grad, inner, sin, sinh, dx, pi, exp, avg, jump, ds, dS, FacetNormal, action, as_matrix, inv, dot
    from firedrake import ds_v, ds_t, ds_b, dS_h, dS_v, HCurlElement, curl
    from firedrake import FiniteElement, TensorProductElement, SpatialCoordinate, interval, quadrilateral, triangle, TensorElement
    from firedrake import HDivElement, split, div, cos, Dx, VectorElement, hexahedron
    from firedrake.petsc import PETSc
    from firedrake import FunctionSpace, MixedFunctionSpace
    from firedrake import Function
    from firedrake import TestFunction, TrialFunction, TestFunctions, TrialFunctions, derivative, CellSize
    from firedrake import norm
    from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver
    from firedrake import Projector
    from firedrake import DumbCheckpoint
    from firedrake import FILE_READ, FILE_UPDATE, FILE_CREATE
    from firedrake import Mesh

    CubeMesh = None

    from firedrake import plot

    def plot_function(func, coords, name):
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
        if len(func.ufl_shape) == 1 and func.ufl_shape[0] == 1:
            plot(func[0], axes=axes)
        else:
            plot(func, axes=axes)
        fig.savefig(name + '.png')
        plt.close('all')

    def get_plotting_spaces(mesh, nquadplot, nquadplot_v=None):
        return None,None,None

    def evaluate_and_store_field(evalspace, opts, field, name, checkpoint):
        return field

if backend == 'themis':
    from utility_meshes import IntervalMesh, PeriodicIntervalMesh, PeriodicSquareMesh, SquareMesh, ExtrudedMesh, CubeMesh, Mesh
    from bcs import DirichletBC
    from constant import Constant
    from ufl import as_vector, grad, inner, sin, sinh, dx, pi, exp, avg, jump, ds, dS, FacetNormal, action, as_matrix, inv, dot
    from ufl import ds_v, ds_t, ds_b, dS_h, dS_v, HCurlElement, curl
    from ufl import FiniteElement, TensorProductElement, SpatialCoordinate, interval, quadrilateral, triangle, TensorElement
    from ufl import HDivElement, split, div, cos, Dx, VectorElement, hexahedron
    from petscshim import PETSc
    from functionspace import FunctionSpace, MixedFunctionSpace
    from function import Function
    from ufl_expr import TestFunction, TrialFunction, TestFunctions, TrialFunctions, derivative, CellSize
    from norms import norm
    from solver import NonlinearVariationalProblem, NonlinearVariationalSolver
    from project import Projector
    from checkpointer import Checkpoint as DumbCheckpoint
    from checkpointer import FILE_READ, FILE_UPDATE, FILE_CREATE
    from mesh import SingleBlockMesh, SingleBlockExtrudedMesh

    from plotting import get_plotting_spaces, evaluate_and_store_field, plot_function

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

def set_mesh_coordinate_order(mesh, ndims, bcs, coordorder, vcoordorder=None):
    vcoordorder = vcoordorder or coordorder

# EXTRUSION CHECKS MIGHT FAIL FOR FIREDRAKE...
# ALSO HOW DO WE PROPERLY HANDLE VARIANT...
# WE NEED THE VARIANT FLAG FOR THEMIS BUT IT WILL FAIL FOR FIREDRAKE!
# Basically something needs to automatically convert variant 'feec' -> None
    if backend == 'themis':
        variant = 'feec'
    if backend == 'firedrake':
        variant = None

    if ndims == 1:
        if bcs[0] == 'nonperiodic':
            celem = FiniteElement("CG", interval, coordorder, variant = variant)
        else:
            celem = FiniteElement("DG", interval, coordorder, variant = variant)
    if ndims == 2 and not mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            celem = FiniteElement("Q", quadrilateral, coordorder, variant = variant)
        else:
            celem = FiniteElement("DQ", quadrilateral, coordorder, variant = variant)
    if ndims == 2 and mesh.extruded:
        if bcs[0] == 'nonperiodic':
            hcelem = FiniteElement("CG", interval, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        else:
            hcelem = FiniteElement("DG", interval, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        celem = TensorProductElement(hcelem,vcelem)
    if ndims == 3 and not mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic' and bcs[2] == 'nonperiodic':
            celem = FiniteElement("Q", hexahedron, coordorder, variant = variant)
        else:
            celem = FiniteElement("DQ", hexahedron, coordorder, variant = variant)
    if ndims == 3 and mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            hcelem = FiniteElement("Q", quadrilateral, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        else:
            hcelem = FiniteElement("DQ", quadrilateral, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        celem = TensorProductElement(hcelem,vcelem)

    velem = VectorElement(celem, dim=ndims)
    coordspace = FunctionSpace(mesh, velem)
    newcoords = Function(coordspace)
    newcoords.interpolate(SpatialCoordinate(mesh))
    return Mesh(newcoords)


def create_box_mesh(cell, nxs, xbcs, coordorder, vcoordorder = None):

    if cell in ['quad', 'tphex']:
        use_quad = True
    if cell in ['tri', 'tptri']:
        use_quad = False

    if cell == 'interval':
        ndims = 1
        if xbcs[0] == 'nonperiodic':
            mesh = IntervalMesh(nxs[0], 1.0)
        if xbcs[0] == 'periodic':
            mesh = PeriodicIntervalMesh(nxs[0], 1.0)

    if cell in ['tri', 'quad']:
        ndims = 2
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            mesh = SquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='y', quadrilateral=use_quad)
    if cell == 'tpquad':
        ndims = 2
        if xbcs[1] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'nonperiodic':
            bmesh = IntervalMesh(nxs[0], 1.0)
        if xbcs[0] == 'periodic':
            bmesh = PeriodicIntervalMesh(nxs[0], 1.0)
        mesh = ExtrudedMesh(bmesh, nxs[1])

    if cell == 'hex':
        ndims = 3
        mesh = CubeMesh(nxs[0], nxs[1], nxs[2], 1.0, xbcs[0], xbcs[1], xbcs[2])
    if cell in ['tphex', 'tptri']:
        ndims = 3
        if xbcs[2] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            bmesh = SquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='y', quadrilateral=use_quad)
        mesh = ExtrudedMesh(bmesh, nxs[2])

    # Required since Firedrake doesn't flag meshes as extruded
    mesh.extruded = False
    if cell in ['tpquad', 'tphex', 'tptri']: mesh.extruded = True

    # Upgrade coordinate order if needed
    if coordorder > 1:
        newmesh = set_mesh_coordinate_order(mesh, ndims, xbcs, coordorder, vcoordorder=vcoordorder)
    else:
        newmesh = mesh

    return newmesh

# FIX THIS
def create_sphere_mesh(cell, rlevel, coordorder, nlevels = None, extrusion_type='uniform', vcoordorder = None):

    if cell == 'quad':
        mesh = CubedSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
    if cell == 'tri':
        mesh = IcosahedralSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
    if cell == 'tphex':
        basemesh = CubedSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
        mesh = ExtrudedMesh(basemesh, nlevels, extrusion_type=extrusion_type)
    if cell == 'tptri':
        basemesh = IcosahedralSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
        mesh = ExtrudedMesh(basemesh, nlevels, extrusion_type=extrusion_type)

    # Upgrade coordinate order if needed
# THIS IS BROKEN
# HOW ARE COORDINATES TREATED FOR SPHERICAL MESHES ANYWAYS
# DEPENDS ON EXTRUSION TYPE...
    if coordorder > 1:
        newmesh = set_mesh_coordinate_order(mesh, ndims, xbcs, coordorder, vcoordorder=vcoordorder)
    else:
        newmesh = mesh
    return newmesh
    
def _quad(order, variant):
    h1elem = FiniteElement("Q", quadrilateral, order, variant=variant)
    l2elem = FiniteElement("DQ", quadrilateral, order-1, variant=variant)
    hdivelem = FiniteElement('RTCF', quadrilateral, order, variant=variant)
    hcurlelem = FiniteElement('RTCE', quadrilateral, order, variant=variant)
    return h1elem, l2elem, hdivelem, hcurlelem

def _tri(order, velocityspace):
    #CGk - RTk-1 - DGk-1 on triangles (order = k)
    if velocityspace == 'RT':
        h1elem = FiniteElement("CG", triangle, order)
        l2elem = FiniteElement('DG', triangle, order-1)
        hdivelem = FiniteElement('RTF', triangle, order) # RT gives spurious inertia-gravity waves
        hcurlelem = FiniteElement('RTE', triangle, order)

    #CGk - BDMk-1 - DGk-2 on triangles (order = k-1)
    if velocityspace == 'BDM':
        h1elem = FiniteElement("CG", triangle, order+1)
        l2elem = FiniteElement("DG", triangle, order-1)
        hdivelem = FiniteElement("BDMF", triangle, order) # BDM gives spurious Rossby waves
        hcurlelem = FiniteElement("BDME", triangle, order)

    #CG2+B3 - BDFM1 - DG1 on triangles
    if velocityspace == 'BDFM':
        if not (order == 2): raise ValueError('BDFM space is only supported for n=2')
        h1elem = FiniteElement("CG", triangle, order) + FiniteElement("Bubble", triangle, order +1)
        l2elem = FiniteElement("DG", triangle, order-1)
        hdivelem = FiniteElement("BDFM", triangle, order) #Note that n=2 is the lowest order element...
        #WHAT IS THE CORRESPONDING HCURL ELEMENT?
        #IS THERE ONE?
        hcurlelem = None
    return h1elem, l2elem, hdivelem, hcurlelem

def _interval(order, variant):
    h1elem = FiniteElement("CG", interval, order, variant=variant)
    l2elem = FiniteElement("DG", interval, order-1, variant=variant)
    return h1elem, l2elem

def _hex(order, variant):
    h1elem = FiniteElement("Q", hexahedron, order, variant=variant)
    l2elem = FiniteElement("DQ", hexahedron, order-1, variant=variant)
    hdivelem = FiniteElement("NCF", hexahedron, order, variant=variant)
    hcurlelem = FiniteElement("NCE", hexahedron, order, variant=variant)
    return h1elem, l2elem, hdivelem, hcurlelem

# ADD SUPPORT FOR CR and SE AS WELL...
# MAYBE? MAYBE JUST DO THEM AS SEPARATE ELEMENTS..
# WHAT ABOUT DG GLL?
def create_complex(cell, velocityspace, variant, order, vorder=None):
    vorder = vorder or order
    if variant == 'none': # this ensures Firedrake works properly
        variant = None

    if cell == 'interval':
        h1elem, l2elem = _interval(order, variant)
        hdivelem = VectorElement(h1elem, dim=1)
        hcurlelem = VectorElement(l2elem, dim=1)
        cpelem = None
    if cell == 'quad':
        h1elem, l2elem, hdivelem, hcurlelem = _quad(order, variant)
        cpelem = None
    if cell == 'tri':
        h1elem, l2elem, hdivelem, hcurlelem = _tri(order, velocityspace)
        cpelem = None
    if cell == 'hex': h1elem, l2elem, hdivelem, hcurlelem = _quad(order, variant)
    if cell in ['tpquad', 'tphex', 'tptri']:
        if cell == 'tpquad':
            h1elem2D, l2elem2D = _interval(order, variant)
            hdivelem2D, hcurlelem2D = _interval(order, variant)
        if cell == 'tphex':
            h1elem2D, l2elem2D, hdivelem2D, hcurlelem2D, _ = _quad(order, variant)
        if cell == 'tptri':
            h1elem2D, l2elem2D, hdivelem2D, hcurlelem2D, _ = _tri(order, velocityspace)
        h1elem1D, l2elem1D = _interval(vorder, variant)
        hdivelem = HDivElement(TensorProductElement(hdivelem2D, l2elem1D)) + HDivElement(TensorProductElement(l2elem2D, h1elem1D))
        hcurlelem = HCurlElement(TensorProductElement(hcurlelem2D, h1elem1D)) + HCurlElement(TensorProductElement(h1elem2D, l2elem1D))
        h1elem = TensorProductElement(h1elem2D, h1elem1D)
        l2elem = TensorProductElement(l2elem2D, l2elem1D)
        cpelem = TensorProductElement(l2elem2D, h1elem1D)

    return {'h1': h1elem, 'l2': l2elem, 'hdiv': hdivelem, 'hcurl': hcurlelem, 'cp': cpelem}

#
# def set_mesh_quadrature(mesh, cell, qdegree = None, vqdegree = None):
#
#     if qdegree:
#         if cell in ['interval', 'quad', 'tri', 'hex']:
#         	mesh.dx = dx(metadata={"quadrature_degree": qdegree})
#         	mesh.dS = dS(metadata={"quadrature_degree": qdegree})
#         	mesh.ds = ds(metadata={"quadrature_degree": qdegree})
# #EVENTUALLY THIS SHOULD ACTUALLY DO SEPARATE DEGREES IN HORIZ AND VERTICAL...
#         if cell in ['tpquad', 'tphex', 'tptri']:
#             vqdegree = vqdegree or qdegree
#         	mesh.dx = dx(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_b = ds_b(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_t = ds_t(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_v = ds_v(metadata={"quadrature_degree": qdegree })
#         	mesh.dS_h = dS_h(metadata={"quadrature_degree": qdegree })
#         	mesh.dS_v = dS_v(metadata={"quadrature_degree": qdegree })
#         	mesh.dS = mesh.dS_h + mesh.dS_v
#         	mesh.ds_tb = mesh.ds_t + mesh.ds_b
#         	mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v
#     else:
#         if cell in ['interval', 'quad', 'tri', 'hex']:
#             mesh.dx = dx
#             mesh.dS = dS
#             mesh.ds = ds
#         if cell in ['tpquad', 'tphex', 'tptri']:
#             mesh.dx = dx
#             mesh.ds_b = ds_b
#             mesh.ds_v = ds_v
#             mesh.ds_t = ds_t
#             mesh.dS_h = dS_h
#             mesh.dS_v = dS_v
#             mesh.dS = mesh.dS_h + mesh.dS_v
#             mesh.ds_tb = mesh.ds_t + mesh.ds_b
#             mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v
#
# def set_boxmesh_operators(mesh, cell):
#
# # ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# # Basically what we need to make Hamiltonian and Hodge Laplacian stuff dimension-agnostic
# 	if cell in ['quad', 'tri', 'tpquad']:
# 		mesh.skewgrad = lambda s : perp(grad(s))
# 		mesh.perp = lambda u : perp(u)
#
#     if cell in ['tpquad']:
# 		mesh.k = as_vector((0,1))
#
#     if cell in ['hex', 'tphex', 'tptri']:
# 		#mesh.skewgrad = lambda s : cross(mesh.k,grad(s))
# 		#mesh.perp = lambda u : cross(mesh.k,u)
#         pass
#
# 	if cell in ['tphex','tptri']:
# 		mesh.k = as_vector((0,0,1))
#
# def make_sphere_normals(x):
# 	R = sqrt(inner(x, x))
# 	rhat_expr = x/R
# 	return rhat_expr
#
# # ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# def set_spheremesh_operators(mesh, cell):
#
#     if cell in ['quad', 'tri']:
#     	xs = SpatialCoordinate(mesh)
#     	mesh.cell_normals = make_sphere_normals(xs)
#     	mesh.perp = lambda u: cross(mesh.cell_normals, u)
#     	mesh.skewgrad = lambda s: cross(mesh.cell_normals, grad(s))
#     	global_normal = as_vector((xs[0],xs[1],xs[2]))
#     	mesh.init_cell_orientations(global_normal)
# # FIX
# 	if cell in ['tphex','tptri']:
#         pass
