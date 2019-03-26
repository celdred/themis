
import numpy as np
from tsfc import compile_expression_at_points as compile_ufl_kernel
import ufl
import themis
import tsfc.kernel_interface.firedrake as kernel_interface
from themis.petscshim import PETSc
from themis.assembly_utils import extract_fields, compile_functional

__all__ = ["Interpolator", "interpolate"]

def interpolate(expr, V, overwrite_pts=None):
    return Interpolator(expr, V, overwrite_pts=overwrite_pts).interpolate()

class Interpolator():

    def __init__(self, expr, V, overwrite_pts=None):
        assert isinstance(expr, ufl.classes.Expr)
        if isinstance(V, themis.Function):
            f = V
            V = f.function_space()
        else:
            f = function.Function(V)

        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [np.prod(fs.ufl_element().value_shape(), dtype=int)
                for fs in V]

        if np.prod(expr.ufl_shape, dtype=int) != sum(dims):
            raise RuntimeError('Expression of length %d required, got length %d'
                               % (sum(dims), np.prod(expr.ufl_shape, dtype=int)))

        if len(V) > 1:
            raise NotImplementedError(
                "UFL expressions for mixed functions are not yet supported.")

        if len(expr.ufl_shape) != len(V.ufl_element().value_shape()):
            raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                               % (len(expr.ufl_shape), len(V.ufl_element().value_shape())))

        if expr.ufl_shape != V.ufl_element().value_shape():
            raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                               % (expr.ufl_shape, V.ufl_element().value_shape()))

        # if expr.ufl_shape != V.shape:
        #    raise ValueError("UFL expression has incorrect shape for interpolation.")

        mesh = V.ufl_domain()
        to_element = V.themis_element()

        if overwrite_pts is None:
            ptsx = np.array(to_element.sub_elements[0][0].spts)
            ptsy = np.array(to_element.sub_elements[0][1].spts)
            ptsz = np.array(to_element.sub_elements[0][2].spts)
        else:
            ptsx,ptsy,ptsz = overwrite_pts
        to_pts = []
        for lx in range(ptsx.shape[0]):
            for ly in range(ptsy.shape[0]):
                for lz in range(ptsz.shape[0]):
                    if mesh.ndim == 1:
                        loc = np.array([ptsx[lx], ])
                    if mesh.ndim == 2:
                        loc = np.array([ptsx[lx], ptsy[ly]])
                    if mesh.ndim == 3:
                        loc = np.array([ptsx[lx], ptsy[ly], ptsz[lz]])
                    to_pts.append(loc)

        ast, oriented, needs_cell_sizes, coefficients, tabulations = compile_ufl_kernel(expr, to_pts, mesh.coordinates, interface=kernel_interface.ExpressionKernelBuilder)

        # process tabulations
        processed_tabulations = []
        for tabname, shape in tabulations:
            splittab = tabname.split('_')

            tabobj = {}
            tabobj['name'] = tabname

            # this matches the string generated in runtime_tabulated.py in FInAT
            # ie variant_order_derivorder_shiftaxis_{d,c}_restrict
            tabobj['variant'] = splittab[1]
            tabobj['order'] = int(splittab[2])
            tabobj['derivorder'] = int(splittab[3])
            tabobj['shiftaxis'] = int(splittab[4])
            if splittab[5] == 'd':
                tabobj['cont'] = 'L2'
            if splittab[5] == 'c':
                tabobj['cont'] = 'H1'
            tabobj['restrict'] = splittab[6]
            tabobj['shape'] = shape

            processed_tabulations.append(tabobj)

        # create InterpolationKernel
        self.kernel = InterpolationKernel(mesh, to_element, to_pts, processed_tabulations, coefficients, ast, V)
        self.field = f

    def interpolate(self):
        InterpolateField(self.field, self.kernel)


class InterpolationKernel():
    def __init__(self, mesh, elem, pts, tabulations, coefficients, ast, functionspace):

        self.integral_type = 'cell'
        self.oriented = False
        self.assemblycompiled = False
        self.interpolate = True
        self.zero = False

        self.pts = pts
        self.elem = elem
        self.tabulations = tabulations
        self.name = ast.name
        self.mesh = mesh
        self.ast = ast
        self.coefficients = coefficients
        self.coefficient_map = np.arange(len(coefficients), dtype=np.int32)
        self.functionspace = functionspace


def InterpolateField(field, interpolationkernel):
    # compile the kernel
    with PETSc.Log.Event('compile'):
        if not interpolationkernel.assemblycompiled:
            compile_functional(interpolationkernel, None, None, interpolationkernel.mesh)

    # scatter fields into local vecs
    with PETSc.Log.Event('extract'):
        extract_fields(interpolationkernel)

    # interpolate
    with PETSc.Log.Event('evaluate'):
        for da, assemblefunc in zip(interpolationkernel.dalist, interpolationkernel.assemblyfunc_list):
            assemblefunc([da, ] + [interpolationkernel.functionspace.get_da(0), field._vector] + interpolationkernel.fieldargs_list, interpolationkernel.constantargs_list)
    field.scatter()
