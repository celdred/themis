
from interop import PETSc, norm, dx, DumbCheckpoint, sin, sinh, inner, grad, FILE_CREATE, exp, cos, perp
from interop import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from interop import TestFunction, DirichletBC, pi
from interop import derivative
from utilities import create_box_mesh, create_complex, adjust_coordinates

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
coordorder = OptDB.getInt('coordorder', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
xbc = OptDB.getString('xbc', 'nonperiodic')  # periodic nonperiodic
ybc = OptDB.getString('ybc', 'nonperiodic')  # periodic nonperiodic
zbc = OptDB.getString('zbc', 'nonperiodic')  # periodic nonperiodic
nx = OptDB.getInt('nx', 16)
ny = OptDB.getInt('ny', 16)
nz = OptDB.getInt('nz', 16)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True)  # periodic nonperiodic
mgd_lowest = OptDB.getBool('mgd_lowest', False)
solntype = OptDB.getString('solntype', 'sin')  # sin exp sinh
c = OptDB.getScalar('c', 0.0)
formorientation = OptDB.getString('formorientation', 'outer')  # outer inner

xbcs = [xbc, ybc, zbc]
nxs = [nx, ny, nz]
lxs = [1.0, 1.0, 1.0]

PETSc.Sys.Print(variant, order, cell, coordorder, xbcs, nxs)

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)

# create mesh and spaces
mesh = create_box_mesh(cell, nxs, xbcs, lxs, coordorder)
elemdict = create_complex(cell, 'rt', variant, order)
adjust_coordinates(mesh, c)


h1 = FunctionSpace(mesh, elemdict['h1'])

hhat = TestFunction(h1)
h = Function(h1, name='h')

# set boundary conditions
bcs = [DirichletBC(h1, 0.0, "on_boundary"), ]
if cell in ['tpquad', 'tphex', 'tptri']:
    bcs.append(DirichletBC(h1, 0.0, "top"))
    bcs.append(DirichletBC(h1, 0.0, "bottom"))


# set rhs/soln
xs = SpatialCoordinate(mesh)

if solntype in ['sinh', 'exp'] and (xbcs[0] == 'periodic' or xbcs[1] == 'periodic' or xbcs[2] == 'periodic'):
    raise ValueError('periodic boundaries requires solntype = sin')
if solntype == 'sinh' and not cell == 'interval':
    raise ValueError('solntype = sinh only works in 1D')

if solntype == 'sinh':
    x = xs[0]
    rhsexpr = x
    solnexpr = x - sinh(x)/sinh(1.)
if solntype == 'exp':
    x = xs[0]
    if cell in 'interval':
        solnexpr = exp(x) * sin(2*pi*x)
        rhsexpr = -(1. - 4.*pi*pi) * exp(x) * sin(2*pi*x) - 4 * pi * exp(x) * cos(2*pi*x) + exp(x) * sin(2*pi*x)
    if cell in ['quad', 'tri', 'tpquad']:
        y = xs[1]
        solnexpr = exp(x+y) * sin(2*pi*x) * sin(2*pi*y)
        rhsexpr = -(1. - 4.*pi*pi) * exp(x+y) * sin(2*pi*x) * sin(2*pi*y) - 4 * pi * exp(x+y) * cos(2*pi*x) * sin(2*pi*y) +\
                  -(1. - 4.*pi*pi) * exp(x+y) * sin(2*pi*x) * sin(2*pi*y) - 4 * pi * exp(x+y) * sin(2*pi*x) * cos(2*pi*y) +\
            exp(x+y) * sin(2*pi*x) * sin(2*pi*y)
    if cell in ['hex', 'tptri', 'tphex']:
        y = xs[1]
        z = xs[2]
        solnexpr = exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
        rhsexpr = -(1. - 4.*pi*pi) * exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z) - 4 * pi * exp(x+y+z) * cos(2*pi*x) * sin(2*pi*y) * sin(2*pi*z) +\
                  -(1. - 4.*pi*pi) * exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z) - 4 * pi * exp(x+y+z) * sin(2*pi*x) * cos(2*pi*y) * sin(2*pi*z) +\
                  -(1. - 4.*pi*pi) * exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z) - 4 * pi * exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * cos(2*pi*z) +\
            exp(x+y+z) * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)

if solntype == 'sin':
    x = xs[0]
    if cell in 'interval':
        rhsexpr = (144. * pi * pi + 4.) * sin(6. * x * pi)
        solnexpr = 4. * sin(6. * x * pi)
    if cell in ['quad', 'tri', 'tpquad']:
        y = xs[1]
        rhsexpr = (80. * pi * pi + 4.) * sin(2. * x * pi) * sin(4. * y * pi)
        solnexpr = 4. * sin(2. * x * pi) * sin(4. * y * pi)
    if cell in ['hex', 'tptri', 'tphex']:
        y = xs[1]
        z = xs[2]
        rhsexpr = (224. * pi * pi + 4.) * sin(2. * x * pi) * sin(4. * y * pi) * sin(6. * z * pi)
        solnexpr = 4. * sin(2. * x * pi) * sin(4. * y * pi) * sin(6. * z * pi)


# create soln and rhs
soln = Function(h1, name='hsoln')

solnproj = Projector(solnexpr, soln, bcs=bcs)  # ,options_prefix= 'masssys_')
solnproj.project()

# Create forms and problem
if formorientation == 'outer' and cell in ['quad', 'tpquad', 'tri']:
    skewgrad = lambda u: perp(grad(u))
    Rlhs = (hhat * h + inner(skewgrad(hhat), skewgrad(h))) * dx
else:
    Rlhs = (hhat * h + inner(grad(hhat), grad(h))) * dx
Rrhs = hhat * rhsexpr * dx

# create solvers
J = derivative(Rlhs - Rrhs, h)

if mgd_lowest:
    from mgd_helpers import lower_form_order
    Jp = lower_form_order(J)
else:
    Jp = J

problem = NonlinearVariationalProblem(Rlhs - Rrhs, h, bcs=bcs, J=J, Jp=Jp)
problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})

# solve system
solver.solve()

# compute norms
l2err = norm(h - soln, norm_type='L2')
h1err = norm(h - soln, norm_type='H1')
l2directerr = norm(h - solnexpr, norm_type='L2')
h1directerr = norm(h - solnexpr, norm_type='H1')
PETSc.Sys.Print(l2err, h1err, l2directerr, h1directerr)

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

# directdiff
directdiff = Function(h1, name='hdirectdiff')
directdiffproj = Projector(h-solnexpr, directdiff)  # options_prefix= 'masssys_'
directdiffproj.project()
checkpoint.store(directdiff)

checkpoint.write_attribute('fields/', 'l2err', l2err)
checkpoint.write_attribute('fields/', 'h1err', h1err)
checkpoint.write_attribute('fields/', 'l2directerr', l2directerr)
checkpoint.write_attribute('fields/', 'h1directerr', h1directerr)

# plot
if plot:
    from interop import plot_function, get_plotting_spaces, evaluate_and_store_field

    scalarevalspace, vectorevalspace, opts = get_plotting_spaces(mesh, nquadplot)

    coordseval = evaluate_and_store_field(vectorevalspace, opts, mesh.coordinates, 'coords', checkpoint)
    heval = evaluate_and_store_field(scalarevalspace, opts, h, 'h', checkpoint)
    hsolneval = evaluate_and_store_field(scalarevalspace, opts, soln, 'hsoln', checkpoint)
    hdiffeval = evaluate_and_store_field(scalarevalspace, opts, diff, 'hdiff', checkpoint)
    hdirectdiffeval = evaluate_and_store_field(scalarevalspace, opts, directdiff, 'hdirectdiff', checkpoint)

    plot_function(heval, coordseval, 'h')
    plot_function(hsolneval, coordseval, 'hsoln')
    plot_function(hdiffeval, coordseval, 'hdiff')
    plot_function(hdirectdiffeval, coordseval, 'hdirectdiff')

checkpoint.close()
