from interop import PETSc, norm, dx, DumbCheckpoint, sin, inner, grad, avg, jump, ds, dS, FacetNormal, action
from interop import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from interop import TestFunction, pi, TrialFunction
from interop import ds_v, ds_t, ds_b, dS_h, dS_v
from interop import derivative
from utilities import create_box_mesh, create_complex, adjust_coordinates
from interop import CellVolume, FacetArea, Constant

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

xbcs = [xbc, ybc, zbc]
nxs = [nx, ny, nz]
lxs = [1.0, 1.0, 1.0]

PETSc.Sys.Print(variant, order, cell, coordorder, xbcs, nxs)

nquadplot_default = order
if variant == 'mgd':
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)


# create mesh and spaces
mesh = create_box_mesh(cell, nxs, xbcs, lxs, coordorder)
elemdict = create_complex(cell, 'rt', variant, order+1)  # must use order + 1 here since this returns FEEC complex compatible L2
adjust_coordinates(mesh, c)

l2 = FunctionSpace(mesh, elemdict['l2'])

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
solnproj = Projector(solnexpr, soln, bcs=[])  # ,options_prefix= 'masssys_'
solnproj.project()

# Create forms and problem
u = TestFunction(l2)
v = TrialFunction(l2)
x = Function(l2, name='x')

if cell in ['tpquad', 'tphex', 'tptri']:
    dIF = dS_v + dS_h
    dEF = ds_v + ds_t + ds_b
else:
    dIF = dS
    dEF = ds

FA = FacetArea(mesh)
CV = CellVolume(mesh)
# ddx = CV/FA
# ddx_avg = (ddx('+') + ddx('-'))/2.
# CELLVOLUME IS BROKEN FOR RUNTIME TABULATED IN TSFC
# Because it is not distinguishing between + and - cells for cell-wise geometric quantities in interior facet integrals
ddx = 1. / nx
ddx_avg = 1. / nx
if variant == 'mgd': # SHOULD BE MORE CLEVER HERE- SCALE BY ORDER FOR SURE!
    alpha = Constant(4.0)
    gamma = Constant(8.0)
else:
    alpha = Constant(1.5 * order * (order + 1.))
    gamma = Constant(3 * order * (order + 1.))
penalty_int = alpha / ddx_avg
penalty_ext = gamma / ddx

n = FacetNormal(mesh)
aV = inner(grad(u), grad(v)) * dx  # volume term
aIF = (inner(jump(u, n), jump(v, n)) * penalty_int - inner(avg(grad(u)), jump(v, n)) - inner(avg(grad(v)), jump(u, n))) * dIF  # interior facet term
aEF = (inner(u, v) * penalty_ext - inner(grad(u), v*n) - inner(grad(v), u*n)) * dEF  # exterior facet term
a = aV + aEF + aIF

Rlhs = action(a, x)
Rrhs = u * rhsexpr * dx

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
    from interop import plot_function, get_plotting_spaces, evaluate_and_store_field

    scalarevalspace, vectorevalspace, opts = get_plotting_spaces(mesh, nquadplot)

    coordseval = evaluate_and_store_field(vectorevalspace, opts, mesh.coordinates, 'coords', checkpoint)
    heval = evaluate_and_store_field(scalarevalspace, opts, x, 'h', checkpoint)
    hsolneval = evaluate_and_store_field(scalarevalspace, opts, soln, 'hsoln', checkpoint)
    hdiffeval = evaluate_and_store_field(scalarevalspace, opts, diff, 'hdiff', checkpoint)
    hdirectdiffeval = evaluate_and_store_field(scalarevalspace, opts, directdiff, 'hdirectdiff', checkpoint)

    plot_function(heval, coordseval, 'h')
    plot_function(hsolneval, coordseval, 'hsoln')
    plot_function(hdiffeval, coordseval, 'hdiff')
    plot_function(hdirectdiffeval, coordseval, 'hdirectdiff')

checkpoint.close()
