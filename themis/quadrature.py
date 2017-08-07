import sympy
from lagrange import gauss_lobatto,lagrange_poly_support,gauss_legendre
import numpy as np

class ThemisQuadrature():
	def __init__(self,qtype,nquadptslist):
		self.exact = False
		if qtype == 'gll':
			quad = GaussLobattoLegendre1D
		if qtype == 'gl':
			quad = GaussLegendre1D
		if qtype == 'gllexact':
			quad = GaussLobattoLegendre1DExact
			self.exact = True
		if qtype == 'glexact':
			quad = GaussLegendre1DExact
			self.exact = True
		#ADD MANY MORE QUAD TYPES HERE!

		if len(nquadptslist) == 1: nquadptslist.append(1)
		if len(nquadptslist) == 2: nquadptslist.append(1)
		self.nquadpts = np.array(nquadptslist,dtype=np.int32)
		self.pts = []
		self.wts = []
		
		for npts in nquadptslist:
			pt,wt = quad(npts)
			self.pts.append(pt)
			self.wts.append(wt)
			
	def get_pts(self):
		return self.pts
	def get_wts(self):
		return self.wts
	def get_nquad(self):
		return self.nquadpts

def GaussLegendre1D(n):
	
	pts = np.array(gauss_legendre(n),dtype=np.float64)
	x = sympy.var('x')
	p = sympy.polys.orthopolys.legendre_poly(n,x) #legendre_poly(n)
	wts = np.zeros((n))
	pd = sympy.diff(p,x)
	pdfunc = sympy.lambdify(x, pd, "numpy")
	diffp = pdfunc(pts)
	wts = 2./((1.-np.square(pts))*np.square(diffp))

	return pts,wts

#This will fail for numerical assembly, since it returns a list instead of a Numpy array
def GaussLegendre1DExact(n):
	if n == 1: 
		pts = [0,]
		wts = [2,]
	if n == 2:
		pts = [-1/sympy.sqrt(3),1/sympy.sqrt(3)]
		wts = [1,1]
	if n == 3:
		pts = [-sympy.sqrt(sympy.Rational(3,5)),0,sympy.sqrt(sympy.Rational(3,5))]
		wts = [sympy.Rational(5,9),sympy.Rational(8,9),sympy.Rational(5,9)]
	if n == 4:
		ap = sympy.sqrt(sympy.Rational(3,7) - sympy.Rational(2,7)*sympy.sqrt(sympy.Rational(6,5)))
		bp = sympy.sqrt(sympy.Rational(3,7) + sympy.Rational(2,7)*sympy.sqrt(sympy.Rational(6,5)))
		aw = sympy.Rational(18,36) + sympy.sqrt(sympy.Rational(30,36*36))
		bw = sympy.Rational(18,36) - sympy.sqrt(sympy.Rational(30,36*36))
		pts = [-bp,-ap,ap,bp]
		wts = [bw,aw,aw,bw] 
		
	if n == 5:
		pts = []
		wts = []

	return pts,wts

#This will fail for numerical assembly, since it returns a list instead of a Numpy array
def GaussLobattoLegendre1DExact(n):
	if n == 1: #THIS IS USEFUL FOR DISPERSION STUFF (FOR GETTING SPTS CORRECT FOR DG0 SPACES) DESPITE NOT ACTUALLY EXISTING
		pts = [0,]
		wts = [2,]
	if n == 2:
		pts = [-1,1]
		wts = [1,1]
	if n == 3:
		pts = [-1,0,1]
		wts = [sympy.Rational(1,3),sympy.Rational(4,3),sympy.Rational(1,3)]
	if n == 4:
		pts = [-1,-1/sympy.sqrt(5),1/sympy.sqrt(5),1]
		wts = [sympy.Rational(1,6),sympy.Rational(5,6),sympy.Rational(5,6),sympy.Rational(1,6)]
	if n == 5:
		ap = sympy.sqrt(sympy.Rational(3,7))
		aw = sympy.Rational(49,90)
		pts = [-1,-ap,0,ap,1]
		wts = [sympy.Rational(1,10),aw,sympy.Rational(32,45),aw,sympy.Rational(1,10)]
		
	if n == 6:

		ap = sympy.sqrt(sympy.Rational(1,3) - 2*sympy.sqrt(sympy.Rational(7,21*21)))
		bp = sympy.sqrt(sympy.Rational(1,3) + 2*sympy.sqrt(sympy.Rational(7,21*21)))
		aw = sympy.Rational(14,30) + sympy.sqrt(sympy.Rational(7,30*30))
		bw = sympy.Rational(14,30) - sympy.sqrt(sympy.Rational(7,30*30))
		pts = [-1,-bp,-ap,ap,bp,1]
		wts = [sympy.Rational(1,15),bw,aw,aw,bw,sympy.Rational(1,15)] 

	if n == 7:
		ap = sympy.sqrt(sympy.Rational(5,11) - 2*sympy.sqrt(sympy.Rational(5,3*11*11)))
		bp = sympy.sqrt(sympy.Rational(5,11) + 2*sympy.sqrt(sympy.Rational(5,3*11*11)))
		aw = sympy.Rational(124,350) + 7*sympy.sqrt(sympy.Rational(15,350*350))
		bw = sympy.Rational(124,350) - 7*sympy.sqrt(sympy.Rational(15,350*350))
		pts = [-1,-bp,-ap,0,ap,bp,1]
		wts = [sympy.Rational(1,21),bw,aw,sympy.Rational(256,525),aw,bw,sympy.Rational(1,21)] 
		
	return pts,wts

def GaussLobattoLegendre1D(n):
	if n == 1: #THIS IS USEFUL FOR DISPERSION STUFF (FOR GETTING SPTS CORRECT FOR DG0 SPACES) DESPITE NOT ACTUALLY EXISTING
		pts = np.array([0.,],dtype=np.float64) 
		wts = np.array([2.,],dtype=np.float64) 
	if n==2:
		pts = np.array([-1.,1.],dtype=np.float64) 
		wts = np.array([1.,1.],dtype=np.float64) 
	if n>=3:	
		pts = np.array(gauss_lobatto(n),dtype=np.float64) 
		wts = np.zeros((n))
		x = sympy.var('x')
		p = sympy.polys.orthopolys.legendre_poly(n-1,x) #legendre_poly(n)
		interior_pts = pts[1:-1]
		p = sympy.polys.orthopolys.legendre_poly(n-1,x) #legendre_poly(n)
		pfunc = sympy.lambdify(x, p, "numpy")
		pnum = pfunc(interior_pts)
		wts[1:-1] = 2./((n*(n-1.))*np.square(pnum))
		wts[0] = 2./(n*(n-1.))
		wts[-1] = 2./(n*(n-1.))
	return pts,wts
