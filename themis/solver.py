
import ufl
import ufl_expr
from petscshim import PETSc
from form import TwoForm, OneForm
# import time

# ADD FIREDRAKE ATTRIBUTION


class NonlinearVariationalProblem():
    """Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None, Jp=None, form_compiler_parameters=None, constant_jacobian=False):
        """
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                         optional, if not supplied then the Jacobian itself
                         will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
                compiler (optional)
        :param dict constant_jacobian: True is J is constant, False if not
        """
        from function import Function

    # Store input UFL forms and solution Function
        self.F = F
        self.Jp = Jp
        self.u = u
        self.bcs = _extract_bcs(bcs)

    # Argument checking
        if not isinstance(self.F, ufl.Form):
            raise TypeError("Provided residual is a '%s', not a Form" % type(self.F).__name__)
        if len(self.F.arguments()) != 1:
            raise ValueError("Provided residual is not a linear form")
        if not isinstance(self.u, Function):
            raise TypeError("Provided solution is a '%s', not a Function" % type(self.u).__name__)

    # Use the user-provided Jacobian. If none is provided, derive
    # the Jacobian from the residual.
        self.J = J or ufl_expr.derivative(F, u)

        if not isinstance(self.J, ufl.Form):
            raise TypeError("Provided Jacobian is a '%s', not a Form" % type(self.J).__name__)
        if len(self.J.arguments()) != 2:
            raise ValueError("Provided Jacobian is not a bilinear form")
        if self.Jp is not None and not isinstance(self.Jp, ufl.Form):
            raise TypeError("Provided preconditioner is a '%s', not a Form" % type(self.Jp).__name__)
        if self.Jp is not None and len(self.Jp.arguments()) != 2:
            raise ValueError("Provided preconditioner is not a bilinear form")

    # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = constant_jacobian

# ADD FIREDRAKE ATTRIBUTION


class NonlinearVariationalSolver():  # solving_utils.ParametersMixin
    """Solves a :class:`NonlinearVariationalProblem`."""

    def __init__(self, problem, **kwargs):
        """
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
                   :class:`.MixedVectorSpaceBasis`) spanning the null
                   space of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
                   make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to
                   specify the near nullspace (for multigrid solvers).
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
                   This should be a dict mapping PETSc options to values.
        :kwarg options_prefix: an optional prefix used to distinguish
                   PETSc options.  If not provided a unique prefix will be
                   created.  Use this option if you want to pass options
                   to the solver from the command line in addition to
                   through the ``solver_parameters`` dict.
        :kwarg pre_jacobian_callback: A user-defined function that will
                   be called immediately before Jacobian assembly. This can
                   be used, for example, to update a coefficient function
                   that has a complicated dependence on the unknown solution.
        :kwarg pre_function_callback: As above, but called immediately
                   before residual assembly

        Example usage of the ``solver_parameters`` option: to set the
        nonlinear solver type to just use a linear solver, use

        .. code-block:: python

                {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values.
        For example:

        .. code-block:: python

                {'snes_monitor': True}

        To use the ``pre_jacobian_callback`` or ``pre_function_callback``
        functionality, the user-defined function must accept the current
        solution as a petsc4py Vec. Example usage is given below:

        .. code-block:: python

                def update_diffusivity(current_solution):
                        with cursol.dat.vec as v:
                                current_solution.copy(v)
                        solve(trial*test*dx == dot(grad(cursol), grad(test))*dx, diffusivity)

                solver = NonlinearVariationalSolver(problem,
                                                                                        pre_jacobian_callback=update_diffusivity)

        """
        assert isinstance(problem, NonlinearVariationalProblem)

        parameters = kwargs.get("solver_parameters")
        # nullspace = kwargs.get("nullspace")
        # nullspace_T = kwargs.get("transpose_nullspace")
        # near_nullspace = kwargs.get("near_nullspace")
        options_prefix = kwargs.get("options_prefix")
        pre_j_callback = kwargs.get("pre_jacobian_callback")
        pre_f_callback = kwargs.get("pre_function_callback")

        # CAN THIS COLLIDE?
        # ONLY AN ISSUE WITH MULTIPLE SOLVERS WITHOUT OPTIONS PREFIXES
        # DOESNT REALLY SHOW UP IN PRACTICE?
        if options_prefix is None:
            options_prefix = str(abs(problem.J.__hash__()))
        OptDB = PETSc.Options()

        # parameters are set in the following order (higher takes priority):
        # 1) command-line
        # 2) solver_parameters keyword argument
        # 3) default (see below)

        # set parameters from solver_parameters
        if not (parameters is None):
            for parameter, value in parameters.items():
                _set_parameter(OptDB, options_prefix + parameter, value)

        # set default parameters
        _set_parameter(OptDB, options_prefix + 'mat_type', 'aij')
        _set_parameter(OptDB, options_prefix + 'pmat_type', 'aij')
        _set_parameter(OptDB, options_prefix + 'ksp_type', 'gmres')
        _set_parameter(OptDB, options_prefix + 'pc_type', 'jacobi')

        # matrix-free
        mat_type = OptDB.getString(options_prefix + 'mat_type')
        pmat_type = OptDB.getString(options_prefix + 'pmat_type')
        matfree = mat_type == "matfree"
        pmatfree = pmat_type == "matfree"

        # No preconditioner by default for matrix-free
        if (problem.Jp is not None and pmatfree) or matfree:
            _set_parameter(OptDB, options_prefix + 'pc_type', 'none')

        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)

        self.snes.setOptionsPrefix(options_prefix)

        self.problem = problem

        # create forms and set function/jacobians and associated assembly functions

        # ADD NULL SPACES
        # ADD FORM COMPILER OPTIONS

        self.Fform = OneForm(problem.F, self.problem.u, bcs=problem.bcs, pre_f_callback=pre_f_callback)
        # WHAT OTHER ARGUMENTS HERE?
        # DOES ABOVE NEED BVALS ARGUMENT?

        self.snes.setFunction(self.Fform.assembleform, self.Fform.vector)

        # EVENTUALLY ADD ABILITY HERE TO HAVE J NON-CONSTANT AND P CONSTANT, ETC.
        if problem.Jp is None:
            self.Jform = TwoForm(problem.J, self.problem.u, mat_type=mat_type, constantJ=problem._constant_jacobian, constantP=problem._constant_jacobian, bcs=problem.bcs, pre_j_callback=pre_j_callback)
            self.snes.setJacobian(self.Jform.assembleform, self.Jform.mat)

        else:
            self.Jform = TwoForm(problem.J, self.problem.u, Jp=problem.Jp, mat_type=mat_type, pmat_type=pmat_type, constantJ=problem._constant_jacobian, constantP=problem._constant_jacobian, bcs=problem.bcs, pre_j_callback=pre_j_callback)
            self.snes.setJacobian(self.Jform.assembleform, self.Jform.mat, self.Jform.pmat)

        # SET NULLSPACE
        # SET NULLSPACE T
        # SET NEAR NULLSPACE
        # nspace = PETSc.NullSpace().create(constant=True)
        # self.form.A.setNullSpace(nspace)
        # ctx.set_nullspace(nullspace, problem.J.arguments()[0].function_space()._ises,transpose=False, near=False)
        # ctx.set_nullspace(nullspace_T, problem.J.arguments()[1].function_space()._ises,transpose=True, near=False)
        # ctx.set_nullspace(near_nullspace, problem.J.arguments()[0].function_space()._ises,transpose=False, near=True)

        self.snes.setFromOptions()

        ismixed = self.problem.u.space.nspaces > 1
        pc = self.snes.getKSP().getPC()
        if ismixed:
            for si1 in range(self.Jform.target.nspaces):
                indices = self.Jform.target.get_field_gis(si1)
                name = str(si1)
                pc.setFieldSplitIS((name, indices))

    def solve(self):
        """Solve the variational problem.

        """

        # Apply the boundary conditions to the initial guess.

        # time1 = time.time()
        for bc in self.problem.bcs:
            bc.apply_vector(self.problem.u._activevector)

        # set solution
        self.snes.setSolution(self.problem.u._activevector)

        self.snes.setUp()
        self.snes.solve(None, self.problem.u._activevector)

        # ADD THIS BACK IN!
        # solving_utils.check_snes_convergence(self.snes)

        # print('full solve',time.time()-time1)
    def destroy(self):
        self.snes.destroy()

# ADD FIREDRAKE ATTRIBUTION


def _extract_bcs(bcs):
    "Extract and check argument bcs"
    from bcs import DirichletBC
    if bcs is None:
        return ()
    try:
        bcs = tuple(bcs)
    except TypeError:
        bcs = (bcs,)
    for bc in bcs:
        if not (isinstance(bc, DirichletBC)):
            raise TypeError("Provided boundary condition is a '%s', not a DirichletBC" % type(bc).__name__)
    return bcs


def _set_parameter(optdb, name, value):
    if not optdb.hasName(name):  # check to make sure parameter has not already been set
        optdb.setValue(name, value)
