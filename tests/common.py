
backend = 'themis'

import matplotlib.pyplot as plt
import numpy as np

if backend == 'firedrake':
    from firedrake import IntervalMesh, PeriodicIntervalMesh, PeriodicSquareMesh, SquareMesh, ExtrudedMesh
    from firedrake import DirichletBC
    from firedrake import Constant
    from firedrake import as_vector, grad, inner, sin, sinh, dx, pi, exp, avg, jump, ds, dS, FacetNormal, action
    from firedrake import ds_v, ds_t, ds_b, dS_h, dS_v, HCurlElement, curl
    from firedrake import FiniteElement, TensorProductElement, SpatialCoordinate, interval, quadrilateral, triangle
    from firedrake import HDivElement, split, div, cos, Dx, VectorElement, hexahedron
    from firedrake.petsc import PETSc
    from firedrake import FunctionSpace, MixedFunctionSpace
    from firedrake import Function
    from firedrake import TestFunction, TrialFunction, TestFunctions, TrialFunctions, derivative
    from firedrake import errornorm
    from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver
    from firedrake import Projector
    from firedrake import DumbCheckpoint
    from firedrake import FILE_READ, FILE_UPDATE, FILE_CREATE

    CubeMesh = None

    from quadrature import ThemisQuadratureNumerical

    class QuadCoefficient():
        def __init__(self, mesh, vartype, varpullback, field, quad, name='test'):
            pass

        def evaluate(self):
            pass

    def store_quad(self, quad, name='test'):
        pass

    DumbCheckpoint.store_quad = store_quad

if backend == 'themis':
    from utility_meshes import IntervalMesh, PeriodicIntervalMesh, PeriodicSquareMesh, SquareMesh, ExtrudedMesh, CubeMesh
    from bcs import DirichletBC
    from constant import Constant
    from ufl import as_vector, grad, inner, sin, sinh, dx, pi, exp, avg, jump, ds, dS, FacetNormal, action
    from ufl import ds_v, ds_t, ds_b, dS_h, dS_v, HCurlElement, curl
    from ufl import FiniteElement, TensorProductElement, SpatialCoordinate, interval, quadrilateral, triangle
    from ufl import HDivElement, split, div, cos, Dx, VectorElement, hexahedron
    from petscshim import PETSc
    from functionspace import FunctionSpace, MixedFunctionSpace
    from function import Function
    from ufl_expr import TestFunction, TrialFunction, TestFunctions, TrialFunctions, derivative
    from norms import errornorm
    from solver import NonlinearVariationalProblem, NonlinearVariationalSolver
    from project import Projector
    from checkpointer import Checkpoint as DumbCheckpoint
    from checkpointer import FILE_READ, FILE_UPDATE, FILE_CREATE

    from evaluate import QuadCoefficient
    from quadrature import ThemisQuadratureNumerical

# universal plotting routine

if backend == 'firedrake':
    from firedrake import plot

    def plot_function(func, funcquad, coordsquad, name):
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
        if len(func.ufl_shape) == 1 and func.ufl_shape[0] == 1:
            plot(func[0], axes=axes)
        else:
            plot(func, axes=axes)
        fig.savefig(name + '.png')
        plt.close('all')

# BROKEN IN PARALLEL
# SHOULD REALLY ACTUALLY JUST READ STUFF FROM THE H5FILE
# THIS IS A DOABLE FIX
if backend == 'themis':
    def plot_function(func, funcquad, coordsquad, name):
        if funcquad.mesh.ndim == 1:
            _plot_function1D(func, funcquad, coordsquad, name)
        if funcquad.mesh.ndim == 2:
            _plot_function2D(func, funcquad, coordsquad, name)
        if funcquad.mesh.ndim == 3:
            _plot_function3D(func, funcquad, coordsquad, name)

    def _plot_function1D(func, funcquad, coordsquad, name):
        funcarr = funcquad.getarray()
        coordsarr = coordsquad.getarray()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        dat = np.ravel(funcarr)
        x = np.ravel(coordsarr[..., 0])
        plt.plot(x, dat)
        fig.savefig(name + '.png')
        plt.close('all')

    def _plot_function2D(func, funcquad, coordsquad, name):
        funcarr = funcquad.getarray()
        coordsarr = coordsquad.getarray()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        x = np.ravel(coordsarr[..., 0])
        y = np.ravel(coordsarr[..., 1])
        if len(funcarr.shape) == len(coordsarr.shape):  # vector quantity
            datu = np.ravel(funcarr[..., 0])
            datv = np.ravel(funcarr[..., 1])
            mag = np.sqrt(np.square(datu) + np.square(datv))
            plt.quiver(x, y, datu, datv, mag)
        else:  # scalar quantity
            dat = np.ravel(funcarr)
            plt.tricontour(x, y, dat, cmap='jet')
            plt.tricontourf(x, y, dat, cmap='jet')
        plt.colorbar()
        fig.savefig(name + '.png')
        plt.close('all')

    # PRETTY BROKEN...
    def _plot_function3D(func, funcquad, coordsquad, name):
        funcarr = funcquad.getarray()
        coordsarr = coordsquad.getarray()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        x = coordsarr[..., 0]
        # y = coordsarr[..., 1]
        z = coordsarr[..., 2]
        # HOW SHOULD WE HANDLE THIS?
        if len(funcarr.shape) == len(coordsarr.shape):  # vector quantity
            pass
        else:  # scalar quantity
            xz_x = np.ravel(x[:, 5, :])
            xz_z = np.ravel(z[:, 5, :])
            xz_dat = np.ravel(funcarr[:, 5, :])
            plt.tricontour(xz_x, xz_z, xz_dat, cmap='jet')
            plt.tricontourf(xz_x, xz_z, xz_dat, cmap='jet')
        plt.colorbar()
        fig.savefig(name + '.xzslice.png')
        plt.close('all')


def create_mesh(nx, ny, nz, ndims, cell, xbcs):
    if cell in ['quad', 'tphex']:
        use_quad = True
    if cell in ['tri', 'tptri']:
        use_quad = False

    if ndims == 1:
        if xbcs[0] == 'nonperiodic':
            mesh = IntervalMesh(nx, 1.0)
        if xbcs[0] == 'periodic':
            mesh = PeriodicIntervalMesh(nx, 1.0)
    if ndims == 2 and cell in ['tri', 'quad']:
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nx, ny, 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            mesh = SquareMesh(nx, ny, 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            mesh = PeriodicSquareMesh(nx, ny, 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nx, ny, 1.0, direction='y', quadrilateral=use_quad)
    if ndims == 2 and cell == 'tpquad':
        if xbcs[1] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'nonperiodic':
            bmesh = IntervalMesh(nx, 1.0)
        if xbcs[0] == 'periodic':
            bmesh = PeriodicIntervalMesh(nx, 1.0)
        mesh = ExtrudedMesh(bmesh, ny)

    if ndims == 3 and cell == 'hex':
        mesh = CubeMesh(nx, ny, nz, 1.0, xbcs[0], xbcs[1], xbcs[2])
    if ndims == 3 and (cell in ['tphex', 'tptri']):
        if xbcs[2] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nx, ny, 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            bmesh = SquareMesh(nx, ny, 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            bmesh = PeriodicSquareMesh(nx, ny, 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nx, ny, 1.0, direction='y', quadrilateral=use_quad)
        mesh = ExtrudedMesh(bmesh, nz)
    return mesh


def create_elems(ndims, cell, variant, order):

    if variant == 'none':
        variant = None
    if ndims == 1:
        h1elem = FiniteElement("CG", interval, order, variant=variant)
        l2elem = FiniteElement("DG", interval, order-1, variant=variant)
        hdivelem = VectorElement(FiniteElement('CG', interval, order, 1, variant=variant), dim=1)
        hcurlelem = VectorElement(FiniteElement('DG', interval, order-1, 1, variant=variant), dim=1)
    if ndims == 2:
        if cell == 'quad':
            h1elem = FiniteElement("Q", quadrilateral, order, variant=variant)
            l2elem = FiniteElement("DQ", quadrilateral, order-1, variant=variant)
            hdivelem = FiniteElement('RTCF', quadrilateral, order, variant=variant)
            hcurlelem = FiniteElement('RTCE', quadrilateral, order, variant=variant)
        if cell == 'tri':
            h1elem = FiniteElement("CG", triangle, order, variant=variant)
            l2elem = FiniteElement('DG', triangle, order-1, variant=variant)
            hdivelem = FiniteElement('RTF', triangle, order, variant=variant)  # EVENTUALLY ADD BDM, ETC TO THIS!
            hcurlelem = FiniteElement('RTE', triangle, order, variant=variant)  # EVENTUALLY ADD BDM, ETC TO THIS!
        if cell == 'tpquad':
            h1elem1D = FiniteElement("CG", interval, order, variant=variant)
            l2elem1D = FiniteElement("DG", interval, order-1, variant=variant)

            l2elem = TensorProductElement(l2elem1D, l2elem1D)
            hdivelem = HDivElement(TensorProductElement(h1elem1D, l2elem1D)) + HDivElement(TensorProductElement(l2elem1D, h1elem1D))
            hcurlelem = HCurlElement(TensorProductElement(l2elem1D, h1elem1D)) + HCurlElement(TensorProductElement(h1elem1D, l2elem1D))
            h1elem = TensorProductElement(h1elem1D, h1elem1D)

    if ndims == 3:
        if cell == 'hex':
            h1elem = FiniteElement("Q", hexahedron, order, variant=variant)
            l2elem = FiniteElement("DQ", hexahedron, order-1, variant=variant)
            hdivelem = FiniteElement("NCF", hexahedron, order, variant=variant)
            hcurlelem = FiniteElement("NCE", hexahedron, order, variant=variant)
        if cell in ['tphex', 'tptri']:

            if cell == 'tphex':
                h1elem2D = FiniteElement("Q", quadrilateral, order, variant=variant)
                l2elem2D = FiniteElement("DQ", quadrilateral, order-1, variant=variant)
                hdivelem2D = FiniteElement('RTCF', quadrilateral, order, variant=variant)
                hcurlelem2D = FiniteElement('RTCE', quadrilateral, order, variant=variant)
            if cell == 'tptri':
                h1elem = FiniteElement("CG", triangle, order, variant=variant)
                l2elem = FiniteElement('DG', triangle, order-1, variant=variant)
                hdivelem = FiniteElement('RTF', triangle, order, variant=variant)  # EVENTUALLY ADD BDM, ETC TO THIS!
                hcurlelem = FiniteElement('RTE', triangle, order, variant=variant)  # EVENTUALLY ADD BDM, ETC TO THIS!

            h1elem1D = FiniteElement("CG", interval, order, variant=variant)
            l2elem1D = FiniteElement("DG", interval, order-1, variant=variant)

            hdivelem = HDivElement(TensorProductElement(hdivelem2D, l2elem1D)) + HDivElement(TensorProductElement(l2elem2D, h1elem1D))
            hcurlelem = HCurlElement(TensorProductElement(hcurlelem2D, h1elem1D)) + HCurlElement(TensorProductElement(h1elem2D, l2elem1D))
            h1elem = TensorProductElement(h1elem2D, h1elem1D)
            l2elem = TensorProductElement(l2elem2D, l2elem1D)

    return h1elem, l2elem, hdivelem, hcurlelem
