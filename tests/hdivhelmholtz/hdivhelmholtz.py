

from common import PETSc,errornorm,dx,DumbCheckpoint,sin,sinh,inner,grad,FILE_CREATE
from common import FunctionSpace,SpatialCoordinate,Function,Projector,NonlinearVariationalProblem,NonlinearVariationalSolver
from common import TestFunction,DirichletBC,pi,TestFunctions
from common import QuadCoefficient,ThemisQuadratureNumerical
from common import split,div,cos,Dx,as_vector,MixedFunctionSpace
from common import create_mesh,create_elems

OptDB = PETSc.Options()
order = OptDB.getInt('order', 1)
simname = OptDB.getString('simname', 'test')
variant = OptDB.getString('variant', 'feec')
xbc = OptDB.getString('xbc', 'nonperiodic') #periodic nonperiodic 
ybc = OptDB.getString('ybc', 'nonperiodic') #periodic nonperiodic
zbc = OptDB.getString('zbc', 'nonperiodic') #periodic nonperiodic
nx = OptDB.getInt('nx', 16) 
ny = OptDB.getInt('ny', 16)
nz = OptDB.getInt('nz', 16)
ndims = OptDB.getInt('ndims', 2)
cell = OptDB.getString('cell', 'quad')
plot = OptDB.getBool('plot', True) #periodic nonperiodic
mgd_lowest = OptDB.getBool('mgd_lowest', False)

nquadplot_default = order
if variant == 'mgd' and order > 1:
    nquadplot_default = 2
nquadplot = OptDB.getInt('nquadplot', nquadplot_default)
xbcs = [xbc,ybc,zbc]	

PETSc.Sys.Print(variant,order,cell,ndims,xbc,ybc,zbc,nx,ny,nz)

#create mesh and spaces
mesh = create_mesh(nx,ny,nz,ndims,cell,xbcs)
h1elem,l2elem,hdivelem,hcurlelem = create_elems(ndims,cell,variant,order)

l2 = FunctionSpace(mesh, l2elem)
hdiv = FunctionSpace(mesh, hdivelem)

mixedspace = MixedFunctionSpace([l2, hdiv])
hhat,uhat = TestFunctions(mixedspace)
xhat = TestFunction(mixedspace)
x = Function(mixedspace,name='x')
h,u  = split(x)

#set boundary conditions
fullbcs = [DirichletBC(mixedspace.sub(1),0.0,"on_boundary"),]
ubcs = [DirichletBC(hdiv,0.0,"on_boundary"),]
if cell in ['tpquad','tphex','tptri']:
    fullbcs.append(DirichletBC(mixedspace.sub(1),0.0,"top"))
    fullbcs.append(DirichletBC(mixedspace.sub(1),0.0,"bottom"))
    ubcs.append(DirichletBC(hdiv,0.0,"top"))
    ubcs.append(DirichletBC(hdiv,0.0,"bottom"))
    
#set rhs/soln
xs = SpatialCoordinate(mesh)

a = 1. / pi
c = 4.
if xbcs[0] == 'periodic':
    fx = sin(2. * xs[0] / a) 
    dfx = 2. / a * cos(2. * xs[0] / a)
    ddfx = -4. / a / a * sin(2. * xs[0] / a)
if xbcs[0] == 'nonperiodic':
    fx = cos(2. * xs[0] / a) 
    dfx = -2. / a * sin(2. * xs[0] / a) 
    ddfx = -4. / a / a * cos(2. * xs[0] / a) 
if ndims >= 2 and xbcs[1] == 'periodic':
    fy = sin(4. * xs[1] / a) 
    dfy = 4. / a * cos(4. * xs[1] / a) 
    ddfy = -16. / a / a * sin(4. * xs[1] / a) 
if ndims >= 2 and xbcs[1] == 'nonperiodic':
    fy = cos(4. * xs[1] / a) 
    dfy = -4. / a * sin(4. * xs[1] / a) 
    ddfy = -16. / a / a * cos(4. * xs[1] / a) 
if ndims >= 3 and xbcs[2] == 'periodic':
    fz = sin(6. * xs[2] / a) 
    dfz = 6. / a * cos(6. * xs[2] / a) 
    ddfz = -36. / a / a * sin(6. * xs[2] / a) 
if ndims >= 3 and xbcs[2] == 'nonperiodic':
    fz = cos(6. * xs[2] / a) 
    dfz = -6. / a * sin(6. * xs[2] / a) 
    ddfz = -36. / a / a * cos(6. * xs[2] / a) 

if ndims == 1:
    hrhsexpr = c * (ddfx + fx)
    hsolnexpr = c * fx
    usolnexpr = c * dfx
    vecsolnexpr = as_vector((1,)) * usolnexpr
if ndims == 2:
    hrhsexpr = c * (ddfx * fy + fx * ddfy + fx * fy)
    hsolnexpr = c * fx * fy
    usolnexpr = c * dfx *  fy
    vsolnexpr = c *  fx * dfy
    vecsolnexpr = as_vector((1,0)) * usolnexpr + as_vector((0,1)) * vsolnexpr
if ndims == 3:
    hrhsexpr = c * (ddfx * fy * fz + fx * ddfy * fz + fx * fy * ddfz + fx * fy * fz)
    hsolnexpr = c * fx * fy * fz
    usolnexpr = c * dfx *  fy * fz
    vsolnexpr = c *  fx * dfy * fz
    wsolnexpr = c *  fx *  fy * dfz
    vecsolnexpr = as_vector((1,0,0)) * usolnexpr + as_vector((0,1,0)) * vsolnexpr + as_vector((0,0,1)) * wsolnexpr

#create soln and rhs
hsoln = Function(l2,name='hsoln')
usoln = Function(hdiv,name='usoln')
hrhs = Function(l2,name='hrhs')

hsolnproj = Projector(hsolnexpr,hsoln,bcs=[]) #,options_prefix= 'masssys_'
usolnproj = Projector(vecsolnexpr,usoln,bcs=ubcs) #,options_prefix= 'masssys_'
hrhsproj = Projector(hrhsexpr,hrhs,bcs=[]) #,options_prefix= 'masssys_'
hsolnproj.project()
usolnproj.project()
hrhsproj.project()


#Create forms and problem
Rlhs = (hhat * h + hhat * div(u) + inner(uhat,u) + div(uhat) * h) * dx
Rrhs = inner(hhat,hrhs) * dx

#create solvers
if mgd_lowest:
    import ufl_expr
    from mgd_helpers import lower_form_order
    J = ufl_expr.derivative(Rlhs- Rrhs, x)
    Jp = lower_form_order(J)
    problem = NonlinearVariationalProblem(Rlhs- Rrhs,x,bcs=fullbcs,Jp=Jp)
else:
    problem = NonlinearVariationalProblem(Rlhs- Rrhs,x,bcs=fullbcs)
    
problem._constant_jacobian = True
solver = NonlinearVariationalSolver(problem,options_prefix = 'linsys_',solver_parameters={'snes_type': 'ksponly'})

#solve system
solver.solve()

#compute norms
hl2err = errornorm(h,hsoln,norm_type='L2')
ul2err = errornorm(u,usoln,norm_type='L2')
uhdiverr = errornorm(u,usoln,norm_type='Hdiv')
PETSc.Sys.Print(hl2err,ul2err,uhdiverr)

#output
checkpoint = DumbCheckpoint(simname,mode=FILE_CREATE)
checkpoint.store(x)
checkpoint.store(hsoln)
checkpoint.store(usoln)
checkpoint.store(mesh.coordinates,name='coords')

hdiff = Function(l2,name='hdiff')
hdiffproj = Projector(h-hsoln,hdiff) #options_prefix= 'masssys_'
hdiffproj.project()
checkpoint.store(hdiff)
udiff = Function(hdiv,name='udiff')
udiffproj = Projector(u-usoln,udiff) #options_prefix= 'masssys_'
udiffproj.project()
checkpoint.store(udiff)

checkpoint.write_attribute('fields/','hl2err',hl2err)
checkpoint.write_attribute('fields/','ul2err',ul2err)
checkpoint.write_attribute('fields/','uhdiverr',uhdiverr)

#evaluate
hsplit,usplit = x.split()
evalquad = ThemisQuadratureNumerical('pascal',[nquadplot,]*ndims)
#THESE SHOULD BE L2 WHEN TSFC/UFL SUPPORTS IT PROPERLY...
hquad = QuadCoefficient(mesh,'scalar','h1',hsplit,evalquad,name='h_quad')
hsolnquad = QuadCoefficient(mesh,'scalar','h1',hsoln,evalquad,name='hsoln_quad')
hdiffquad = QuadCoefficient(mesh,'scalar','h1',hdiff,evalquad,name='hdiff_quad')
uquad = QuadCoefficient(mesh,'vector','hdiv',usplit,evalquad,name='u_quad')
usolnquad = QuadCoefficient(mesh,'vector','hdiv',usoln,evalquad,name='usoln_quad')
udiffquad = QuadCoefficient(mesh,'vector','hdiv',udiff,evalquad,name='udiff_quad')
coordsquad = QuadCoefficient(mesh,'vector','h1',mesh.coordinates,evalquad,name='coords_quad')
hquad.evaluate()
hsolnquad.evaluate()
hdiffquad.evaluate()
uquad.evaluate()
usolnquad.evaluate()
udiffquad.evaluate()
coordsquad.evaluate()
checkpoint.store_quad(hquad)
checkpoint.store_quad(hsolnquad)
checkpoint.store_quad(hdiffquad)
checkpoint.store_quad(uquad)
checkpoint.store_quad(usolnquad)
checkpoint.store_quad(udiffquad)
checkpoint.store_quad(coordsquad)

checkpoint.close()

if plot:
    from common import plot_function
    plot_function(hsplit,hquad,coordsquad,'h')
    plot_function(hsoln,hsolnquad,coordsquad,'hsoln')
    plot_function(hdiff,hdiffquad,coordsquad,'hdiff')
    plot_function(usplit,uquad,coordsquad,'u')
    plot_function(usoln,usolnquad,coordsquad,'usoln')
    plot_function(udiff,udiffquad,coordsquad,'udiff')
