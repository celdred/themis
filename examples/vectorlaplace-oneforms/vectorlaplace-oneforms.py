

from interop import PETSc, errornorm, dx, DumbCheckpoint, sin, inner, grad, FILE_CREATE, Dx, norm
from interop import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from interop import TestFunction, pi, TestFunctions
from interop import split, div, cos, as_vector, MixedFunctionSpace, curl
from utilities import create_box_mesh, create_complex, adjust_coordinates

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
coordorder = OptDB.getInt('coordorder', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
velocityspace = OptDB.getString('velocityspace', 'rt')
xbc = OptDB.getString('xbc', 'nonperiodic')  # periodic nonperiodic
ybc = OptDB.getString('ybc', 'nonperiodic')  # periodic nonperiodic
zbc = OptDB.getString('zbc', 'nonperiodic')  # periodic nonperiodic
nx = OptDB.getInt('nx', 16)
ny = OptDB.getInt('ny', 16)
nz = OptDB.getInt('nz', 16)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True)
mgd_lowest = OptDB.getBool('mgd_lowest', False)
c = OptDB.getScalar('c', 0.0)

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)

xbcs = [xbc, ybc, zbc]
nxs = [nx, ny, nz]
lxs = [1.0, 1.0, 1.0]

PETSc.Sys.Print(variant, velocityspace, order, cell, coordorder, xbcs, nxs)

# create mesh and spaces
mesh = create_box_mesh(cell, nxs, xbcs, lxs, coordorder)
elemdict = create_complex(cell, velocityspace, variant, order)
# THIS IS BROKEN!
# I am not sure the definitions for u/v/etc. work on a distorted mesh...
adjust_coordinates(mesh, c)

if cell == 'interval':
    raise ValueError("must be in 2D or 3D")

h1 = FunctionSpace(mesh, elemdict['h1'])
hcurl = FunctionSpace(mesh, elemdict['hcurl'])

mixedspace = MixedFunctionSpace([h1, hcurl])
hhat, uhat = TestFunctions(mixedspace)
xhat = TestFunction(mixedspace)
x = Function(mixedspace, name='x')
h, u = split(x)

# set rhs/soln
xs = SpatialCoordinate(mesh)

k = 2
p = 3
m = 4
sinx = sin(k * pi * xs[0])
cosx = cos(k * pi * xs[0])
siny = sin(p * pi * xs[1])
cosy = cos(p * pi * xs[1])
if cell in ['hex', 'tptri', 'tphex']:
    sinz = sin(m * pi * xs[2])
    cosz = cos(m * pi * xs[2])

if cell in ['quad', 'tri', 'tpquad']:
    urhsexpr = pi * pi * (k * k + p * p) * sinx * cosy
    vrhsexpr = pi * pi * (k * k + p * p) * cosx * siny
    vecrhsexpr = as_vector((urhsexpr, vrhsexpr))
    usolnexpr = sinx * cosy
    vsolnexpr = cosx * siny
    vecsolnexpr = as_vector((usolnexpr, vsolnexpr))
    hsolnexpr = - cosx * cosy * (k * pi + p * pi)
if cell in ['hex', 'tptri', 'tphex']:
    urhsexpr = pi * pi * (k * k + p * p) * sinx * cosy
    vrhsexpr = pi * pi * (p * p + m * m) * siny * cosz
    wrhsexpr = pi * pi * (k * k + m * m) * sinz * cosx
    vecrhsexpr = as_vector((urhsexpr, vrhsexpr, wrhsexpr))
    usolnexpr = sinx * cosy
    vsolnexpr = siny * cosz
    wsolnexpr = sinz * cosx
    vecsolnexpr = as_vector((usolnexpr, vsolnexpr, wsolnexpr))
    hsolnexpr = - cosx * cosy * cosz * (k * pi + p * pi + m * pi)

# HOW DO BCS WORK FOR THIS PROBLEM?
# THIS IS THE INTERESTING BIT...
# Right now we are treating only natural BCs...
# Work on more interesting ones!

# create soln and rhs
hsoln = Function(h1, name='hsoln')
usoln = Function(hcurl, name='usoln')

hsolnproj = Projector(hsolnexpr, hsoln, bcs=[])
usolnproj = Projector(vecsolnexpr, usoln, bcs=[])
hsolnproj.project()
usolnproj.project()

# Create forms and problem
if cell in ['quad', 'tri', 'tpquad']:
    rot = lambda u: -Dx(u[0],1) + Dx(u[1],0)
    Rlhs = (hhat * h - inner(grad(hhat), u) + inner(uhat, grad(h)) + inner(rot(uhat), rot(u))) * dx
if cell in ['hex', 'tptri', 'tphex']:
    Rlhs = (hhat * h - inner(grad(hhat), u) + inner(uhat, grad(h)) - inner(curl(uhat), curl(u))) * dx
Rrhs = inner(uhat, vecrhsexpr) * dx

# create solvers
problem = NonlinearVariationalProblem(Rlhs - Rrhs, x, bcs=[])
problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})

# solve system
solver.solve()

# compute norms
hl2err = norm(h - hsoln, norm_type='L2')
hh1err = norm(h - hsoln, norm_type='H1')
ul2err = norm(u - usoln, norm_type='L2')
uhcurlerr = norm(u - usoln, norm_type='Hcurl')
hl2directerr = norm(h - hsolnexpr, norm_type='L2')
hh1directerr = norm(h - hsolnexpr, norm_type='H1')
ul2directerr = norm(u - vecsolnexpr, norm_type='L2')
uhcurldirecterr = norm(u - vecsolnexpr, norm_type='Hcurl')
PETSc.Sys.Print(hl2err, hh1err, ul2err, uhcurlerr, hl2directerr, hh1directerr, ul2directerr, uhcurldirecterr)

# output
checkpoint = DumbCheckpoint(simname, mode=FILE_CREATE)
checkpoint.store(x)
checkpoint.store(hsoln)
checkpoint.store(usoln)
checkpoint.store(mesh.coordinates, name='coords')

hdiff = Function(h1, name='hdiff')
hdiffproj = Projector(h-hsoln, hdiff)
hdiffproj.project()
checkpoint.store(hdiff)
udiff = Function(hcurl, name='udiff')
udiffproj = Projector(u-usoln, udiff)
udiffproj.project()
checkpoint.store(udiff)

hdirectdiff = Function(h1, name='hdirectdiff')
hdirectdiffproj = Projector(h-hsolnexpr, hdirectdiff)
hdirectdiffproj.project()
checkpoint.store(hdirectdiff)
udirectdiff = Function(hcurl, name='udirectdiff')
udirectdiffproj = Projector(u-vecsolnexpr, udirectdiff)
udirectdiffproj.project()
checkpoint.store(udirectdiff)

checkpoint.write_attribute('fields/', 'hl2err', hl2err)
checkpoint.write_attribute('fields/', 'hh1err', hh1err)
checkpoint.write_attribute('fields/', 'ul2err', ul2err)
checkpoint.write_attribute('fields/', 'uhcurlerr', uhcurlerr)
checkpoint.write_attribute('fields/', 'hl2directerr', hl2directerr)
checkpoint.write_attribute('fields/', 'hh1directerr', hh1directerr)
checkpoint.write_attribute('fields/', 'ul2directerr', ul2directerr)
checkpoint.write_attribute('fields/', 'uhcurldirecterr', uhcurldirecterr)

if plot:
    from interop import plot_function, get_plotting_spaces, evaluate_and_store_field

    # This is needed due to faulty handling of SplitFunctions within Themis KernelExpressionBuilder
    # Need to refactor split(f), f.split(), W.sub(), split(W), etc.
    hsplit, usplit = x.split()
    honly = Function(h1, name='h')
    uonly = Function(hcurl, name='u')
    honly.assign(hsplit)
    uonly.assign(usplit)

    scalarevalspace, vectorevalspace, opts = get_plotting_spaces(mesh, nquadplot)

    coordseval = evaluate_and_store_field(vectorevalspace, opts, mesh.coordinates, 'coords', checkpoint)
    heval = evaluate_and_store_field(scalarevalspace, opts, honly, 'h', checkpoint)
    hsolneval = evaluate_and_store_field(scalarevalspace, opts, hsoln, 'hsoln', checkpoint)
    hdiffeval = evaluate_and_store_field(scalarevalspace, opts, hdiff, 'hdiff', checkpoint)
    hdirectdiffeval = evaluate_and_store_field(scalarevalspace, opts, hdirectdiff, 'hdirectdiff', checkpoint)
    ueval = evaluate_and_store_field(vectorevalspace, opts, uonly, 'u', checkpoint)
    usolneval = evaluate_and_store_field(vectorevalspace, opts, usoln, 'usoln', checkpoint)
    udiffeval = evaluate_and_store_field(vectorevalspace, opts, udiff, 'udiff', checkpoint)
    udirectdiffeval = evaluate_and_store_field(vectorevalspace, opts, udirectdiff, 'udirectdiff', checkpoint)

    plot_function(heval, coordseval, 'h')
    plot_function(hsolneval, coordseval, 'hsoln')
    plot_function(hdiffeval, coordseval, 'hdiff')
    plot_function(hdirectdiffeval, coordseval, 'hdirectdiff')
    plot_function(ueval, coordseval, 'u')
    plot_function(usolneval, coordseval, 'usoln')
    plot_function(udiffeval, coordseval, 'udiff')
    plot_function(udirectdiffeval, coordseval, 'udirectdiff')

checkpoint.close()
