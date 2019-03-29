

from interop import PETSc, dx, DumbCheckpoint, inner, grad, exp
from interop import FunctionSpace, Function, NonlinearVariationalProblem, NonlinearVariationalSolver
from interop import TrialFunction, TestFunction, DirichletBC, Constant
from interop import derivative
from utilities import create_box_mesh, create_complex, adjust_coordinates

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
coordorder = OptDB.getInt('coordorder', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
lambdaa = Constant(OptDB.getScalar('lambda', 5.0))
nx = OptDB.getInt('nx', 16)
ny = OptDB.getInt('ny', 16)
nz = OptDB.getInt('nz', 16)
ndims = OptDB.getInt('ndims', 2)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True)
mgd_lowest = OptDB.getBool('mgd_lowest', False)
c = OptDB.getScalar('c', 0.0)

xbcs = ['nonperiodic', 'nonperiodic', 'nonperiodic']
nxs = [nx, ny, nz]
lxs = [1.0, 1.0, 1.0]

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)

PETSc.Sys.Print(variant, order, cell, coordorder, nxs)

# create mesh and spaces

mesh = create_box_mesh(cell, nxs, xbcs, lxs, coordorder)
elemdict = create_complex(cell, 'rt', variant, order)
adjust_coordinates(mesh, c)

h1 = FunctionSpace(mesh, elemdict['h1'])

hhat = TestFunction(h1)
h = TrialFunction(h1)

# set boundary conditions
bcs = [DirichletBC(h1, 0.0, "on_boundary"), ]
if cell in ['tpquad', 'tphex', 'tptri']:
    bcs.append(DirichletBC(h1, 0.0, "top"))
    bcs.append(DirichletBC(h1, 0.0, "bottom"))

# Create forms and problem
x = Function(h1, name='x')
R = (-hhat * lambdaa * exp(x) + inner(grad(hhat), grad(x))) * dx  # degree=(order*2+1)
# J = (-hhat * lambdaa * exp(x) * h + inner(grad(hhat), grad(h))) * dx  # (degree=(order*2+1))
J = derivative(R, x)

# create solvers
if mgd_lowest:
    from mgd_helpers import lower_form_order
    Jp = lower_form_order(J)
else:
    Jp = J

problem = NonlinearVariationalProblem(R, x, J=J, Jp=Jp, bcs=bcs)
solver = NonlinearVariationalSolver(problem, options_prefix='nonlinsys_')

# solve system
solver.solve()

# output
checkpoint = DumbCheckpoint(simname)
checkpoint.store(x)
checkpoint.store(mesh.coordinates)

# plot
if plot:
    from interop import plot_function, get_plotting_spaces, evaluate_and_store_field

    scalarevalspace, vectorevalspace, opts = get_plotting_spaces(mesh, nquadplot)

    coordseval = evaluate_and_store_field(vectorevalspace, opts, mesh.coordinates, 'coords', checkpoint)
    eval = evaluate_and_store_field(scalarevalspace, opts, x, 'x', checkpoint)

    plot_function(eval, coordseval, 'x')

checkpoint.close()
