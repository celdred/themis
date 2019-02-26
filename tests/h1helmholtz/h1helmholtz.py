
from common import PETSc, errornorm, dx, DumbCheckpoint, sin, sinh, inner, grad, FILE_CREATE
from common import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from common import TestFunction, DirichletBC, pi
from common import QuadCoefficient, ThemisQuadratureNumerical
from common import create_mesh, create_elems

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
xbc = OptDB.getString('xbc', 'nonperiodic')  # periodic nonperiodic
ybc = OptDB.getString('ybc', 'nonperiodic')  # periodic nonperiodic
zbc = OptDB.getString('zbc', 'nonperiodic')  # periodic nonperiodic
nx = OptDB.getInt('nx', 16)
ny = OptDB.getInt('ny', 16)
nz = OptDB.getInt('nz', 16)
ndims = OptDB.getInt('ndims', 2)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True)  # periodic nonperiodic
mgd_lowest = OptDB.getBool('mgd_lowest', False)
use_sinh = OptDB.getBool('use_sinh', True)


PETSc.Sys.Print(variant, order, cell, ndims, xbc, ybc, zbc, nx, ny, nz)

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)
xbcs = [xbc, ybc, zbc]


# create mesh and spaces
mesh = create_mesh(nx, ny, nz, ndims, cell, xbcs)
h1elem, l2elem, hdivelem, hcurlelem = create_elems(ndims, cell, variant, order)

h1 = FunctionSpace(mesh, h1elem)

hhat = TestFunction(h1)
h = Function(h1, name='h')

# set boundary conditions
bcs = [DirichletBC(h1, 0.0, "on_boundary"), ]
if cell in ['tpquad', 'tphex', 'tptri']:
    bcs.append(DirichletBC(h1, 0.0, "top"))
    bcs.append(DirichletBC(h1, 0.0, "bottom"))


# set rhs/soln
xs = SpatialCoordinate(mesh)

a = 1. / pi
scale = 1
if ndims == 1:
    x = xs[0]
    if use_sinh and xbcs[0] == 'nonperiodic':
        rhsexpr = x
        solnexpr = x - sinh(x)/sinh(1.)
        scale = -1
    else:
        rhsexpr = (-144. / a / a + 4.) * sin(6. * x / a)
        solnexpr = 4. * sin(6. * x / a)
if ndims == 2:
    x = xs[0]
    y = xs[1]
    rhsexpr = (-80. / a / a + 4.) * sin(2. * x / a) * sin(4. * y / a)
    solnexpr = 4. * sin(2. * x / a) * sin(4. * y / a)
if ndims == 3:
    x = xs[0]
    y = xs[1]
    z = xs[2]
    rhsexpr = (-224. / a / a + 4.) * sin(2. * x / a) * sin(4. * y / a) * sin(6. * z / a)
    solnexpr = 4. * sin(2. * x / a) * sin(4. * y / a) * sin(6. * z / a)

# create soln and rhs
soln = Function(h1, name='hsoln')
rhs = Function(h1, name='rhs')

solnproj = Projector(solnexpr, soln, bcs=bcs)  # ,options_prefix= 'masssys_')
rhsproj = Projector(rhsexpr, rhs, bcs=bcs)  # options_prefix= 'masssys_'
solnproj.project()
rhsproj.project()
rhsproj.project()

# Create forms and problem
# Degree estimation for tensor product elements with more than 2 components is broken, hence the need for the below
Rlhs = (hhat * h - scale*inner(grad(hhat), grad(h))) * dx  # degree=(order*2+1)
Rrhs = hhat * rhs * dx  # degree=(order*2+1)

# create solvers
if mgd_lowest:
    import ufl_expr
    from mgd_helpers import lower_form_order
    J = ufl_expr.derivative(Rlhs - Rrhs, h)
    Jp = lower_form_order(J)
    problem = NonlinearVariationalProblem(Rlhs - Rrhs, h, bcs=bcs, Jp=Jp)
else:
    problem = NonlinearVariationalProblem(Rlhs - Rrhs, h, bcs=bcs)

problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})


# solve system
solver.solve()

# compute norms
l2err = errornorm(h, soln, norm_type='L2')
h1err = errornorm(h, soln, norm_type='H1')
PETSc.Sys.Print(l2err, h1err)

# output
checkpoint = DumbCheckpoint(simname, mode=FILE_CREATE)
checkpoint.store(h)
checkpoint.store(soln)
checkpoint.store(mesh.coordinates)

# diff
diff = Function(h1, name='hdiff')
diffproj = Projector(h-soln, diff)  # options_prefix= 'masssys_'
diffproj.project()
checkpoint.store(diff)

checkpoint.write_attribute('fields/', 'l2err', l2err)
checkpoint.write_attribute('fields/', 'h1err', h1err)

# evaluate
evalquad = ThemisQuadratureNumerical('pascal', [nquadplot, ]*ndims)
hquad = QuadCoefficient(mesh, 'scalar', 'h1', h, evalquad, name='h_quad')
solnquad = QuadCoefficient(mesh, 'scalar', 'h1', soln, evalquad, name='hsoln_quad')
diffquad = QuadCoefficient(mesh, 'scalar', 'h1', diff, evalquad, name='hdiff_quad')
coordsquad = QuadCoefficient(mesh, 'vector', 'h1', mesh.coordinates, evalquad, name='coords_quad')
hquad.evaluate()
solnquad.evaluate()
diffquad.evaluate()
coordsquad.evaluate()
checkpoint.store_quad(hquad)
checkpoint.store_quad(solnquad)
checkpoint.store_quad(diffquad)
checkpoint.store_quad(coordsquad)

checkpoint.close()

# plot
if plot:
    from common import plot_function
    plot_function(h, hquad, coordsquad, 'h')
    plot_function(soln, solnquad, coordsquad, 'hsoln')
    plot_function(diff, diffquad, coordsquad, 'hdiff')
