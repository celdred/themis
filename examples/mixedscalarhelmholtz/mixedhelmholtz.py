

from interop import PETSc, norm, dx, DumbCheckpoint, sin, inner, FILE_CREATE
from interop import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from interop import TestFunction, DirichletBC, pi, TestFunctions
from interop import split, div, cos, as_vector, MixedFunctionSpace, Dx
from interop import derivative
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
formorientation = OptDB.getString('formorientation', 'outer')  # outer inner

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
adjust_coordinates(mesh, c)

l2 = FunctionSpace(mesh, elemdict['l2'])
if formorientation == 'inner' and cell in ['quad', 'tpquad', 'tri']:
    hdiv = FunctionSpace(mesh, elemdict['hcurl'])
else:
    hdiv = FunctionSpace(mesh, elemdict['hdiv'])

mixedspace = MixedFunctionSpace([l2, hdiv])
hhat, uhat = TestFunctions(mixedspace)
xhat = TestFunction(mixedspace)
x = Function(mixedspace, name='x')
h, u = split(x)

# set boundary conditions
fullbcs = [DirichletBC(mixedspace.sub(1), 0.0, "on_boundary"), ]
ubcs = [DirichletBC(hdiv, 0.0, "on_boundary"), ]
if cell in ['tpquad', 'tphex', 'tptri']:
    fullbcs.append(DirichletBC(mixedspace.sub(1), 0.0, "top"))
    fullbcs.append(DirichletBC(mixedspace.sub(1), 0.0, "bottom"))
    ubcs.append(DirichletBC(hdiv, 0.0, "top"))
    ubcs.append(DirichletBC(hdiv, 0.0, "bottom"))

# set rhs/soln
xs = SpatialCoordinate(mesh)

c = 4.
if xbcs[0] == 'periodic':
    fx = sin(2. * xs[0] * pi)
    dfx = 2. * pi * cos(2. * xs[0] * pi)
    ddfx = -4. * pi * pi * sin(2. * xs[0] * pi)
if xbcs[0] == 'nonperiodic':
    fx = cos(2. * xs[0] * pi)
    dfx = -2. * pi * sin(2. * xs[0] * pi)
    ddfx = -4. * pi * pi * cos(2. * xs[0] * pi)
if cell in ['quad', 'tri', 'tpquad', 'hex', 'tptri', 'tphex'] and xbcs[1] == 'periodic':
    fy = sin(4. * xs[1] * pi)
    dfy = 4. * pi * cos(4. * xs[1] * pi)
    ddfy = -16. * pi * pi * sin(4. * xs[1] * pi)
if cell in ['quad', 'tri', 'tpquad', 'hex', 'tptri', 'tphex'] and xbcs[1] == 'nonperiodic':
    fy = cos(4. * xs[1] * pi)
    dfy = -4. * pi * sin(4. * xs[1] * pi)
    ddfy = -16. * pi * pi * cos(4. * xs[1] * pi)
if cell in ['hex', 'tptri', 'tphex'] and xbcs[2] == 'periodic':
    fz = sin(6. * xs[2] * pi)
    dfz = 6. * pi * cos(6. * xs[2] * pi)
    ddfz = -36. * pi * pi * sin(6. * xs[2] * pi)
if cell in ['hex', 'tptri', 'tphex'] and xbcs[2] == 'nonperiodic':
    fz = cos(6. * xs[2] * pi)
    dfz = -6. * pi * sin(6. * xs[2] * pi)
    ddfz = -36. * pi * pi * cos(6. * xs[2] * pi)

if cell == 'interval':
    hrhsexpr = c * (-ddfx + fx)
    hsolnexpr = c * fx
    usolnexpr = -c * dfx
    vecsolnexpr = as_vector((usolnexpr,))
if cell in ['quad', 'tri', 'tpquad']:
    hrhsexpr = c * (-ddfx * fy - fx * ddfy + fx * fy)
    hsolnexpr = c * fx * fy
    if formorientation == 'inner':
        usolnexpr = c * fx * dfy
        vsolnexpr = -c * dfx * fy
    if formorientation == 'outer':
        usolnexpr = -c * dfx * fy
        vsolnexpr = -c * fx * dfy
    vecsolnexpr = as_vector((usolnexpr, vsolnexpr))
if cell in ['hex', 'tptri', 'tphex']:
    hrhsexpr = c * (-ddfx * fy * fz - fx * ddfy * fz - fx * fy * ddfz + fx * fy * fz)
    hsolnexpr = c * fx * fy * fz
    usolnexpr = -c * dfx * fy * fz
    vsolnexpr = -c * fx * dfy * fz
    wsolnexpr = -c * fx * fy * dfz
    vecsolnexpr = as_vector((usolnexpr, vsolnexpr, wsolnexpr))

# create soln and rhs
hsoln = Function(l2, name='hsoln')
usoln = Function(hdiv, name='usoln')

hsolnproj = Projector(hsolnexpr, hsoln, bcs=[])
usolnproj = Projector(vecsolnexpr, usoln, bcs=ubcs)
hsolnproj.project()
usolnproj.project()


# Create forms and problem
if formorientation == 'inner' and cell in ['quad', 'tpquad', 'tri']:
    rot = lambda u: -Dx(u[0], 1) + Dx(u[1], 0)
    Rlhs = (hhat * h + hhat * rot(u) + inner(uhat, u) - rot(uhat) * h) * dx
else:
    Rlhs = (hhat * h + hhat * div(u) + inner(uhat, u) - div(uhat) * h) * dx
Rrhs = inner(hhat, hrhsexpr) * dx

# create solvers
J = derivative(Rlhs - Rrhs, x)
if mgd_lowest:
    from mgd_helpers import lower_form_order
    Jp = lower_form_order(J)
else:
    Jp = J
problem = NonlinearVariationalProblem(Rlhs - Rrhs, x, bcs=fullbcs, J=J, Jp=Jp)

problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})

# solve system
solver.solve()

# compute norms
hl2err = norm(h - hsoln, norm_type='L2')
ul2err = norm(u - usoln, norm_type='L2')
if formorientation == 'inner' and cell in ['quad', 'tpquad', 'tri']:
    uhdiverr = norm(u - usoln, norm_type='Hcurl')
else:
    uhdiverr = norm(u - usoln, norm_type='Hdiv')
hl2directerr = norm(h - hsolnexpr, norm_type='L2')
ul2directerr = norm(u - vecsolnexpr, norm_type='L2')
if formorientation == 'inner' and cell in ['quad', 'tpquad', 'tri']:
    uhdivdirecterr = norm(u - vecsolnexpr, norm_type='Hcurl')
else:
    uhdivdirecterr = norm(u - vecsolnexpr, norm_type='Hdiv')
PETSc.Sys.Print(hl2err, ul2err, uhdiverr, hl2directerr, ul2directerr, uhdivdirecterr)

# output
checkpoint = DumbCheckpoint(simname, mode=FILE_CREATE)
checkpoint.store(x)
checkpoint.store(hsoln)
checkpoint.store(usoln)
checkpoint.store(mesh.coordinates, name='coords')

hdiff = Function(l2, name='hdiff')
hdiffproj = Projector(h-hsoln, hdiff)
hdiffproj.project()
checkpoint.store(hdiff)
udiff = Function(hdiv, name='udiff')
udiffproj = Projector(u-usoln, udiff)
udiffproj.project()
checkpoint.store(udiff)

hdirectdiff = Function(l2, name='hdirectdiff')
hdirectdiffproj = Projector(h-hsolnexpr, hdirectdiff)
hdirectdiffproj.project()
checkpoint.store(hdirectdiff)
udirectdiff = Function(hdiv, name='udirectdiff')
udirectdiffproj = Projector(u-vecsolnexpr, udirectdiff)
udirectdiffproj.project()
checkpoint.store(udirectdiff)

checkpoint.write_attribute('fields/', 'hl2err', hl2err)
checkpoint.write_attribute('fields/', 'ul2err', ul2err)
checkpoint.write_attribute('fields/', 'uhdiverr', uhdiverr)
checkpoint.write_attribute('fields/', 'hl2directerr', hl2directerr)
checkpoint.write_attribute('fields/', 'ul2directerr', ul2directerr)
checkpoint.write_attribute('fields/', 'uhdivdirecterr', uhdivdirecterr)

if plot:
    from interop import plot_function, get_plotting_spaces, evaluate_and_store_field

    # This is needed due to faulty handling of SplitFunctions within Themis KernelExpressionBuilder
    # Need to refactor split(f), f.split(), W.sub(), split(W), etc.
    hsplit, usplit = x.split()
    honly = Function(l2, name='h')
    uonly = Function(hdiv, name='u')
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
