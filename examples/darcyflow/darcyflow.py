

from common import PETSc, norm, dx, DumbCheckpoint, sin, inner, FILE_CREATE
from common import FunctionSpace, SpatialCoordinate, Function, Projector, NonlinearVariationalProblem, NonlinearVariationalSolver
from common import TestFunction, DirichletBC, pi, TestFunctions
from common import split, div, cos, as_vector, MixedFunctionSpace
from common import create_box_mesh, create_complex, adjust_coordinates
from common import derivative, Constant, FacetNormal, ds, dS, as_matrix, inv, dot, TensorElement
from common import np
import time

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
coordorder = OptDB.getInt('coordorder', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
velocityspace = OptDB.getString('variant', 'rt')
nx = OptDB.getInt('nx', 16)
ny = OptDB.getInt('ny', 16)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True)  # periodic nonperiodic
mgd_lowest = OptDB.getBool('mgd_lowest', False)
kappa_value = OptDB.getString('kappa_value', 'identity')
kappa_type = OptDB.getString('kappa_type', 'ufl')

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)
xbcs = ['nonperiodic', 'nonperiodic']
nxs = [nx, ny]

PETSc.Sys.Print(variant, order, cell, coordorder, nxs)

start = time.time()

# create mesh and spaces
mesh = create_box_mesh(cell, nxs, xbcs, coordorder)
elemdict = create_complex(cell, velocityspace, variant, order)

xs = SpatialCoordinate(mesh)
newcoords = Function(mesh.coordinates.function_space(), name='newcoords')
xlist = [xs[0] + 0.03 * sin(3./2.*pi*(2*xs[0]-1.))*sin(3./2.*pi*(2*xs[1]-1.)), xs[1] - 0.04 * sin(3./2.*pi*(2*xs[0]-1.))*sin(3./2.*pi*(2*xs[1]-1.))]
newcoords.interpolate(as_vector(xlist))
mesh.coordinates.assign(newcoords)

l2 = FunctionSpace(mesh, elemdict['l2'])
hdiv = FunctionSpace(mesh, elemdict['hdiv'])

mixedspace = MixedFunctionSpace([l2, hdiv])
hhat, uhat = TestFunctions(mixedspace)
xhat = TestFunction(mixedspace)
xn = Function(mixedspace, name='x')
h, u = split(xn)

end = time.time()
PETSc.Sys.Print('made spaces/functions',end-start)
start = time.time()

# set rhs/soln
x,y = SpatialCoordinate(mesh)


if kappa_value == 'identity':
    alpha,beta,gamma,delta = 1.,0.,0.,1.
    alphax,betax,gammay,deltay = 0.,0.,0.,0.

if kappa_value == 'constant':
    alpha,beta,gamma,delta = 0.8,0.2,0.2,0.6
    alphax,betax,gammay,deltay = 0.,0.,0.,0.

if kappa_value == 'variable':
    alpha = (x+1)*(x+1) + y*y
    beta = sin(x*y)
    gamma = sin(x*y)
    delta = (x+1)*(x+1)
    alphax = 2*(x+1)
    betax = y*cos(x*y)
    gammay = x*cos(x*y)
    deltay = 0.

hsolnexpr = x*x*x * y*y*y*y + x*x + sin(x*y) * cos(y)
px = 3*x*x *y*y*y*y + 2*x + y*cos(x*y) * cos(y)
py = x*x*x * 4*y*y*y + x*cos(x*y) * cos(y) - sin(x*y) * sin(y)
pxx = 6*x *y*y*y*y + 2 - y*y*sin(x*y) * cos(y)
pyy = x*x*x * 12*y*y - x*x*sin(x*y) * cos(y) - x*cos(x*y) * sin(y) - x*cos(x*y) * sin(y) - sin(x*y) * cos(y)
pxy = 12*x*x*y*y*y - x*y*sin(x*y)*cos(y) + cos(x*y) * cos(y) - y*cos(x*y) * sin(y)
hrhsexpr = -(alphax * px + alpha * pxx + betax * py + beta * pxy +
            gammay * px + gamma *pxy + deltay * py + delta * pyy)
usolnexpr = -( alpha * px + beta * py )
vsolnexpr = -( gamma * px + delta * py )
vecsolnexpr = as_vector((usolnexpr, vsolnexpr))

gexpr = hsolnexpr
kappamat = [[alpha,beta],[gamma,delta]]

if kappa_type == 'field':
    highelemdict = create_complex(cell, velocityspace, variant, order +1 )
    l2tensorelem = TensorElement(highelemdict['l2'], shape = (2,2))
    l2tensorspace = FunctionSpace(mesh, l2tensorelem)
    kappa = Function(l2tensorspace, name = 'kappa')
    kappaproj = Projector(as_matrix(kappamat), kappa, bcs=[])
    kappaproj.project()
elif kappa_type == 'const':
    kappa = Constant(kappamat)
elif kappa_type == 'ufl':
    kappa = as_matrix(kappamat)


# create soln and rhs
hsoln = Function(l2, name='hsoln')
usoln = Function(hdiv, name='usoln')

hsolnproj = Projector(hsolnexpr, hsoln, bcs=[])  # ,options_prefix= 'masssys_'
usolnproj = Projector(vecsolnexpr, usoln, bcs=[])  # ,options_prefix= 'masssys_'
hsolnproj.project()
usolnproj.project()

end = time.time()
PETSc.Sys.Print('projected',end-start)
start = time.time()

# Create forms and problem
kappainv = inv(kappa)
n = FacetNormal(mesh)
Rlhs = (hhat * div(u) + inner(uhat, dot(kappainv,u)) - div(uhat) * h) * dx
Rrhs = inner(hhat, hrhsexpr) * dx - inner(gexpr,inner(uhat,n)) * ds

# create solvers
J = derivative(Rlhs - Rrhs, xn)
if mgd_lowest:
    from mgd_helpers import lower_form_order
    Jp = lower_form_order(J)
else:
    Jp = J
problem = NonlinearVariationalProblem(Rlhs - Rrhs, xn, bcs=[], J=J, Jp=Jp)

problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem, options_prefix='linsys_', solver_parameters={'snes_type': 'ksponly'})

end = time.time()
PETSc.Sys.Print('made solver',end-start)
start = time.time()

# solve system
solver.solve()

end = time.time()
PETSc.Sys.Print('solved',end-start)
start = time.time()

# compute norms
hl2err = norm(h - hsoln, norm_type='L2')
ul2err = norm(u - usoln, norm_type='L2')
uhdiverr = norm(u - usoln, norm_type='Hdiv')
hl2directerr = norm(h - hsolnexpr, norm_type='L2')
ul2directerr = norm(u - vecsolnexpr, norm_type='L2')
uhdivdirecterr = norm(u - vecsolnexpr, norm_type='Hdiv')
PETSc.Sys.Print(hl2err, ul2err, uhdiverr, hl2directerr, ul2directerr, uhdivdirecterr)

# output
checkpoint = DumbCheckpoint(simname, mode=FILE_CREATE)
checkpoint.store(xn)
checkpoint.store(hsoln)
checkpoint.store(usoln)
checkpoint.store(mesh.coordinates, name='coords')

hdiff = Function(l2, name='hdiff')
hdiffproj = Projector(h-hsoln, hdiff)  # options_prefix= 'masssys_'
hdiffproj.project()
checkpoint.store(hdiff)
udiff = Function(hdiv, name='udiff')
udiffproj = Projector(u-usoln, udiff)  # options_prefix= 'masssys_'
udiffproj.project()
checkpoint.store(udiff)

hdirectdiff = Function(l2, name='hdirectdiff')
hdirectdiffproj = Projector(h-hsolnexpr, hdirectdiff)  # options_prefix= 'masssys_'
hdirectdiffproj.project()
checkpoint.store(hdirectdiff)
udirectdiff = Function(hdiv, name='udirectdiff')
udirectdiffproj = Projector(u-vecsolnexpr, udirectdiff)  # options_prefix= 'masssys_'
udirectdiffproj.project()
checkpoint.store(udirectdiff)
checkpoint.write_attribute('fields/', 'hl2err', hl2err)
checkpoint.write_attribute('fields/', 'ul2err', ul2err)
checkpoint.write_attribute('fields/', 'uhdiverr', uhdiverr)
checkpoint.write_attribute('fields/', 'hl2directerr', hl2directerr)
checkpoint.write_attribute('fields/', 'ul2directerr', ul2directerr)
checkpoint.write_attribute('fields/', 'uhdivdirecterr', uhdivdirecterr)

end = time.time()
PETSc.Sys.Print('norms and output',end-start)
start = time.time()

if plot:
    from common import plot_function, get_plotting_spaces, evaluate_and_store_field

    # This is needed due to faulty handling of SplitFunctions within Themis KernelExpressionBuilder
    # Need to refactor split(f), f.split(), W.sub(), split(W), etc.
    hsplit,usplit = xn.split()
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

    plot_function(heval,coordseval,'h')
    plot_function(hsolneval,coordseval,'hsoln')
    plot_function(hdiffeval,coordseval,'hdiff')
    plot_function(hdirectdiffeval,coordseval,'hdirectdiff')
    plot_function(ueval,coordseval,'u')
    plot_function(usolneval,coordseval,'usoln')
    plot_function(udiffeval,coordseval,'udiff')
    plot_function(udirectdiffeval,coordseval,'udirectdiff')

checkpoint.close()

end = time.time()
PETSc.Sys.Print('plotted',end-start)
start = time.time()
