from common import PETSc, norm, dx, DumbCheckpoint, sin, inner, grad, avg, jump, ds, dS, FacetNormal, action
from common import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from common import TestFunction, pi, TrialFunction
from common import QuadCoefficient, ThemisQuadratureNumerical, ds_v, ds_t, ds_b, dS_h, dS_v
from common import create_mesh, create_elems
from common import derivative

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
c = OptDB.getScalar('c', 0.0)
coordorder = OptDB.getInt('coordorder', 1)

PETSc.Sys.Print(variant, order, cell, ndims, xbc, ybc, zbc, nx, ny, nz)

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)
xbcs = [xbc, ybc, zbc]


# create mesh and spaces
mesh = create_mesh(nx, ny, nz, ndims, cell, xbcs, c, coordorder)
h1elem, l2elem, hdivelem, hcurlelem = create_elems(ndims, cell, variant, order+1)  # must use order + 1 here since this returns FEEC complex compatible L2

l2 = FunctionSpace(mesh, l2elem)

# set rhs/soln
xs = SpatialCoordinate(mesh)

a = 1. / pi
if ndims == 1:
    x = xs[0]
    rhsexpr = (144. / a / a) * sin(6. * x / a)
    solnexpr = 4. * sin(6. * x / a)
if ndims == 2:
    x = xs[0]
    y = xs[1]
    rhsexpr = (80. / a / a) * sin(2. * x / a) * sin(4. * y / a)
    solnexpr = 4. * sin(2. * x / a) * sin(4. * y / a)
if ndims == 3:
    x = xs[0]
    y = xs[1]
    z = xs[2]
    rhsexpr = (224. / a / a) * sin(2. * x / a) * sin(4. * y / a) * sin(6. * z / a)
    solnexpr = 4. * sin(2. * x / a) * sin(4. * y / a) * sin(6. * z / a)

soln = Function(l2, name='soln')
rhs = Function(l2, name='rhs')
solnproj = Projector(solnexpr, soln, bcs=[])  # ,options_prefix= 'masssys_'
rhsproj = Projector(rhsexpr, rhs, bcs=[])  # ,options_prefix= 'masssys_'
solnproj.project()
rhsproj.project()

# Create forms and problem
u = TestFunction(l2)
v = TrialFunction(l2)
x = Function(l2, name='x')

n = FacetNormal(mesh)
# THIS ASSUMES A UNIFORM GRID, SHOULD BE MORE CLEVER...
ddx = 1. / nx
if variant == 'mgd':
    penalty = 1. * (1. + 1.) / ddx
else:
    penalty = order * (order + 1.) / ddx

if cell in ['tpquad', 'tphex', 'tptri']:
    dIF = dS_v + dS_h
    dEF = ds_v + ds_t + ds_b
else:
    dIF = dS
    dEF = ds

aV = inner(grad(u), grad(v)) * dx  # volume term
aIF = (inner(jump(u, n), jump(v, n)) * penalty - inner(avg(grad(u)), jump(v, n)) - inner(avg(grad(v)), jump(u, n))) * dIF  # interior facet term
aEF = (u*v * penalty - inner(grad(u), v*n) - inner(grad(v), u*n)) * dEF  # exterior facet term
a = aV + aEF + aIF

Rlhs = action(a, x)
Rrhs = u * rhs * dx

# create solvers
J = derivative(Rlhs - Rrhs, x)
problem = NonlinearVariationalProblem(Rlhs - Rrhs, x, J=J, Jp=J, bcs=[])
problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})

# solve system
solver.solve()

# compute norms
l2err = norm(x - soln, norm_type='L2')
l2directerr = norm(x - solnexpr, norm_type='L2')
PETSc.Sys.Print(l2err, l2directerr)

# output
checkpoint = DumbCheckpoint(simname)
checkpoint.store(x)
checkpoint.store(soln)
checkpoint.store(mesh.coordinates)


diff = Function(l2, name='diff')
diffproj = Projector(x-soln, diff)  # options_prefix= 'masssys_'
diffproj.project()
checkpoint.store(diff)

directdiff = Function(l2, name='directdiff')
directdiffproj = Projector(x-solnexpr, directdiff)  # options_prefix= 'masssys_'
directdiffproj.project()
checkpoint.store(directdiff)

checkpoint.write_attribute('fields/', 'l2err', l2err)
checkpoint.write_attribute('fields/', 'l2directerr', l2directerr)


# plot
if plot:
    # evaluate
    evalquad = ThemisQuadratureNumerical('pascal', [nquadplot, ])
    # SWAP TO L2 ONCE TSFC/UFL SUPPORT IT...
    xquad = QuadCoefficient(mesh, 'scalar', 'h1', x, evalquad, name='x_quad')
    solnquad = QuadCoefficient(mesh, 'scalar', 'h1', soln, evalquad, name='soln_quad')
    diffquad = QuadCoefficient(mesh, 'scalar', 'h1', diff, evalquad, name='diff_quad')
    directdiffquad = QuadCoefficient(mesh, 'scalar', 'h1', directdiff, evalquad, name='directdiff_quad')
    coordsquad = QuadCoefficient(mesh, 'vector', 'h1', mesh.coordinates, evalquad, name='coords_quad')
    xquad.evaluate()
    solnquad.evaluate()
    diffquad.evaluate()
    directdiffquad.evaluate()
    coordsquad.evaluate()
    checkpoint.store_quad(xquad)
    checkpoint.store_quad(solnquad)
    checkpoint.store_quad(diffquad)
    checkpoint.store_quad(directdiffquad)
    checkpoint.store_quad(coordsquad)

    from common import plot_function
    plot_function(x, xquad, coordsquad, 'x')
    plot_function(soln, solnquad, coordsquad, 'soln')
    plot_function(diff, diffquad, coordsquad, 'diff')
    plot_function(directdiff, directdiffquad, coordsquad, 'directdiff')

checkpoint.close()
