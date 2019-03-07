
from function import Function
import numpy as np
from tsfc import compile_expression_at_points as compile_ufl_kernel

class Interpolator():
    def __init__(self):
        pass

compile_at_points()

class Interpolator():

    def __init__(self, expr, V):
        assert isinstance(expr, ufl.classes.Expr)
        if isinstance(V, Function):
            f = V
            V = f.function_space()
        else:
            f = Function(V)

        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [numpy.prod(fs.ufl_element().value_shape(), dtype=int)
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
                               
        if expr.ufl_shape != V.shape:
            raise ValueError("UFL expression has incorrect shape for interpolation.")
            
        mesh = V.ufl_domain()
        to_element = V.themis_element()
        
        # ACTUALLY GENERATE THIS CORRECTLY IE TP IT
        # HERE WE CAN SAFELY ASSUME CI = 0 SINCE WE DONT INTERPOLATE INTO ENRICHED ELEMENTS...
        to_pts = to_element._spts[ci][direc]
        ast, oriented, needs_cell_sizes, coefficients = compile_ufl_kernel(expr, to_pts, mesh.coordinates)
        
        # DOES THIS UFL KERNEL REQUIRE TABULATIONS? SEEMS LIKELY...
        # IE IT SHOULD!
        # FIX FROM HERE DOWN...
            
    def interpolate(self):

        #FIX THIS BIT
        # actually run the kernel!
        return






