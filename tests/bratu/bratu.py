

from common import PETSc, dx, DumbCheckpoint, inner, grad, exp
from common import FunctionSpace, Function, NonlinearVariationalProblem, NonlinearVariationalSolver
from common import TrialFunction, TestFunction, DirichletBC, Constant
from common import QuadCoefficient, ThemisQuadratureNumerical
from common import create_mesh, create_elems

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
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

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)

PETSc.Sys.Print(variant, order, cell, ndims, nx, ny, nz)

# create mesh and spaces
xbcs = ['nonperiodic', 'nonperiodic', 'nonperiodic']
mesh = create_mesh(nx, ny, nz, ndims, cell, xbcs)
h1elem, l2elem, hdivelem, hcurlelem = create_elems(ndims, cell, variant, order)

h1 = FunctionSpace(mesh, h1elem)

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
J = (-hhat * lambdaa * exp(x) * h + inner(grad(hhat), grad(h))) * dx  # (degree=(order*2+1))

# create solvers
if mgd_lowest:
    from mgd_helpers import lower_form_order
    Jp = lower_form_order(J)
    problem = NonlinearVariationalProblem(R, x, J=J, Jp=Jp, bcs=bcs)
else:
    problem = NonlinearVariationalProblem(R, x, J=J, bcs=bcs)

solver = NonlinearVariationalSolver(problem, options_prefix='nonlinsys_')

# solve system
solver.solve()

# output
checkpoint = DumbCheckpoint(simname)
checkpoint.store(x)
checkpoint.store(mesh.coordinates)

# evaluate
evalquad = ThemisQuadratureNumerical('pascal', [nquadplot, ]*ndims)
xquad = QuadCoefficient(mesh, 'scalar', 'h1', x, evalquad, name='x_quad')
coordsquad = QuadCoefficient(mesh, 'vector', 'h1', mesh.coordinates, evalquad, name='coords_quad')
xquad.evaluate()
coordsquad.evaluate()
checkpoint.store_quad(xquad)
checkpoint.store_quad(coordsquad)

checkpoint.close()

# plot
if plot:
    from common import plot_function
    plot_function(x, xquad, coordsquad, 'x')
