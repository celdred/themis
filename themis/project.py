
import ufl
from themis import ufl_expr
from themis.function import Function
from themis.solver import NonlinearVariationalProblem, NonlinearVariationalSolver

# NEEDS FIREDRAKE ATTRIBUTION

__all__ = ["Projector",]

class Projector():
    """
    A projector projects a UFL expression into a function space
    and places the result in a function from that function space,
    allowing the solver to be reused. Projection reverts to an assign
    operation if ``v`` is a :class:`.Function` and belongs to the same
    function space as ``v_out``.

    :arg v: the :class:`ufl.Expr` or
             :class:`.Function` to project or a
             list of :class:`ufl.Expr`
    :arg v_out: :class:`.Function` to put the result in
    :arg solver_parameters: parameters to pass to the solver used when
             projecting.
    """

    def __init__(self, v, v_out, bcs=None, options_prefix=None, solver_parameters=None):
        if isinstance(v, list) or isinstance(v, tuple):  # type(v) == type([]) or type(v) == type(()):
            vlist = v
        else:
            vlist = [v, ]
        for vin in vlist:
            if not isinstance(vin, (ufl.core.expr.Expr, Function, int, float)):
                raise ValueError("Can only project UFL expression or Functions not '%s'" % type(vin))
        if (len(vlist) > 1) and (not (len(vlist) == v_out.function_space().nspaces)):
            raise ValueError("List of expr or Functions must be the same size as underlying mixed space in v_out")

        self._same_fspace = (isinstance(v, Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out

        if not self._same_fspace:
            V = v_out.function_space()

            p = ufl_expr.TestFunction(V)

            lhs = ufl.inner(p, v_out)*ufl.dx

            if len(vlist) == 1:
                rhs = ufl.inner(p, v)*ufl.dx
            else:
                plist = ufl_expr.TestFunctions(V)
                rhs = 0
                for i, p in enumerate(plist):
                    if not (vlist[i] == 0):
                        rhs = rhs + ufl.inner(p, vlist[i])*ufl.dx

            if solver_parameters is None:
                solver_parameters = {}

            solver_parameters.setdefault("snes_type", "ksponly")

            problem = NonlinearVariationalProblem(lhs - rhs, v_out, bcs=bcs, constant_jacobian=True)
            self.solver = NonlinearVariationalSolver(problem, options_prefix=options_prefix, solver_parameters=solver_parameters)

    def project(self):
        """
        Apply the projection.
        """
        if self._same_fspace:
            self.v_out.assign(self.v)
        else:
            self.solver.solve()
