

from ufl import FiniteElement, TensorProductElement, VectorElement, interval, quadrilateral, hexahedron
from themis.functionspace import FunctionSpace
from themis.function import Function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

__all__ = ["get_plotting_spaces", "evaluate_and_store_field", "plot_function"]

def Pascal1D(n):
    pts = np.linspace(-1., 1., 2*n+1)
    pts = pts[1:-1:2]
    return 0.5 * pts + 0.5

def get_plotting_spaces(mesh, nquadplot, nquadplot_v=None):
    nquadplot_v = nquadplot_v or nquadplot
    if mesh.ndim == 1: scalarelem = FiniteElement("DG", interval, nquadplot-1, variant='feec')
    if mesh.ndim == 2 and not mesh.extruded: scalarelem = FiniteElement("DQ", quadrilateral, nquadplot-1, variant='feec')
    if mesh.ndim == 2 and mesh.extruded: scalarelem = TensorProductElement(FiniteElement("DG", interval, nquadplot-1, variant='feec'), FiniteElement("DG", interval, nquadplot_v-1, variant='feec'))
    if mesh.ndim == 3 and not mesh.extruded: scalarelem = FiniteElement("DG", hexahedron, nquadplot-1, variant='feec')
    if mesh.ndim == 3 and mesh.extruded: scalarelem = TensorProductElement(FiniteElement("DG", quadrilateral, nquadplot-1, variant='feec'), FiniteElement("DG", interval, nquadplot_v-1, variant='feec'))
    vectorelem = VectorElement(scalarelem, dim=mesh.ndim)
    scalarevalspace = FunctionSpace(mesh, scalarelem)
    vectorevalspace = FunctionSpace(mesh, vectorelem)
    sptsl2 = Pascal1D(nquadplot)
    sptsl2v = Pascal1D(nquadplot_v)
    middle = Pascal1D(1)
    if mesh.ndim == 1: opts = [sptsl2,middle,middle]
    if mesh.ndim == 2 and not mesh.extruded: opts = [sptsl2,sptsl2,middle]
    if mesh.ndim == 2 and mesh.extruded: opts = [sptsl2,sptsl2v,middle]
    if mesh.ndim == 3 and not mesh.extruded: opts = [sptsl2,sptsl2,sptsl2]
    if mesh.ndim == 3 and mesh.extruded: opts = [sptsl2,sptsl2,sptsl2v]

    return scalarevalspace, vectorevalspace, opts

def evaluate_and_store_field(evalspace, opts, field, name, checkpoint):
    evalfield = Function(evalspace, name = name + 'eval')
    evalfield.interpolate(field, overwrite_pts=opts)
    checkpoint.store(evalfield)
    return evalfield

# BROKEN IN PARALLEL
# SHOULD REALLY ACTUALLY JUST READ STUFF FROM THE H5FILE
# THIS IS A DOABLE FIX

def plot_function(func, coords, name):
    da = func.function_space().get_da(0)
    funcarr = da.getVecArray(func._vector)[:]
    da = coords.function_space().get_da(0)
    coordsarr = da.getVecArray(coords._vector)[:]
    if func.function_space()._mesh.ndim == 1:
        _plot_function1D(funcarr, np.expand_dims(coordsarr,axis=-1), name)
    if func.function_space()._mesh.ndim == 2:
        _plot_function2D(funcarr, coordsarr, name)
    if func.function_space()._mesh.ndim == 3:
        _plot_function3D(funcarr, coordsarr, name)

def _plot_function1D(funcarr, coordsarr, name):
    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    dat = np.ravel(funcarr)
    x = np.ravel(coordsarr[..., 0])
    plt.plot(x, dat)
    fig.savefig(name + '.png')
    plt.close('all')

def _plot_function2D(funcarr, coordsarr, name):
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
        triang = tri.Triangulation(x, y)
        mask = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.05)
        triang.set_mask(mask)
        dat = np.ravel(funcarr)
        plt.tripcolor(triang, dat, cmap='jet', shading='gouraud')
        #plt.tricontour(x, y, dat, cmap='jet')
        #plt.tricontourf(x, y, dat, cmap='jet')
    plt.colorbar()
    fig.savefig(name + '.png')
    plt.close('all')

# PRETTY BROKEN...
# slices?
# some fancy sort of 3D thing?
def _plot_function3D(funcarr, coordsarr, name):
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
