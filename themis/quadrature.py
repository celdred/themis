import sympy
from lagrange import gauss_lobatto, gauss_legendre,lagrange_poly_support
import numpy as np
from finat.quadrature import TensorProductQuadratureRule


def rescale_pts(pts):
    return 0.5 * pts + 0.5


def rescale_wts(wts):
    return wts * 0.5


def rescale_pts_exact(pt):
    return pt * sympy.Rational(1, 2) + sympy.Rational(1, 2)


def rescale_wts_exact(wt):
    return wt * sympy.Rational(1, 2)


class ThemisQuadrature():

    def get_pts(self):
        return self.pts

    def get_wts(self):
        return self.wts

    def get_nquad(self):
        return self.nquadpts


class ThemisQuadratureNumerical(ThemisQuadrature):
    def __init__(self, qtype, nquadptslist):

        self.exact = False
        self.finatquad = None
        if qtype == 'gll':
            quad = GaussLobattoLegendre1D
        if qtype == 'gl':
            quad = GaussLegendre1D
        if qtype == 'newtoncotesopen':
            quad = NewtonCotesOpen1D
        if qtype == 'pascal':
            quad = Pascal1D
        # ADD MANY MORE QUAD TYPES HERE!

        if len(nquadptslist) == 1:
            nquadptslist.append(1)
        if len(nquadptslist) == 2:
            nquadptslist.append(1)
        self.nquadpts = np.array(nquadptslist, dtype=np.int32)
        self.pts = []
        self.wts = []
        for npts in nquadptslist:
            pt, wt = quad(npts)
            self.pts.append(rescale_pts(pt))
            self.wts.append(rescale_wts(wt))


class ThemisQuadratureExact(ThemisQuadrature):
    def __init__(self, qtype, nquadptslist):

        self.exact = True
        self.finatquad = None
        if qtype == 'gllexact':
            quad = GaussLobattoLegendre1DExact
        if qtype == 'glexact':
            quad = GaussLegendre1DExact

        if len(nquadptslist) == 1:
            nquadptslist.append(1)
        if len(nquadptslist) == 2:
            nquadptslist.append(1)
        self.nquadpts = np.array(nquadptslist, dtype=np.int32)
        self.pts = []
        self.wts = []
        for npts in nquadptslist:
            pt, wt = quad(npts)
            for i in range(len(pt)):
                pt[i] = rescale_pts_exact(pt[i])
                wt[i] = rescale_wts_exact(wt[i])
            self.pts.append(pt)
            self.wts.append(wt)


class ThemisQuadratureFinat(ThemisQuadrature):

    def __init__(self, finatquad):
        self.exact = False
        self.finatquad = finatquad
        nquadptslist = []
        self.pts = []
        self.wts = []

        if isinstance(finatquad, TensorProductQuadratureRule):
            for ps, qr in zip(finatquad.point_set.factors, finatquad.factors):
                self.pts.append(ps.points[:, 0])
                self.wts.append(qr.weights)
                nquadptslist.append(ps.points.shape[0])
        else:
            # print('fq',finatquad.point_set.points)
            # print(finatquad)
            # print(finatquad.point_set.points.shape)
            # try:
            self.pts.append(finatquad.point_set.points[:, 0])
            nquadptslist.append(finatquad.point_set.points.shape[0])
            # except:
            self.pts.append(np.array([0., ]))
            nquadptslist.append(1)
            self.wts.append(finatquad.weights)

        if len(self.pts) == 1:
            pt, wt = GaussLobattoLegendre1D(1)
            self.pts.append(rescale_pts(pt))
            self.pts.append(rescale_pts(pt))
            self.wts.append(rescale_wts(wt))
            self.wts.append(rescale_wts(wt))
            nquadptslist.append(1)
            nquadptslist.append(1)
        elif len(self.pts) == 2:
            pt, wt = GaussLobattoLegendre1D(1)
            self.pts.append(rescale_pts(pt))
            self.wts.append(rescale_wts(wt))
            nquadptslist.append(1)
        self.nquadpts = np.array(nquadptslist, dtype=np.int32)


##### ALL OF THESE QUADRATURE RULES ARE DEFINED ON THE INTERVAL -1,1 ########

def NewtonCotesOpen1D(n):

	pts = np.linspace(-1.,1.,n+2)
	pts = pts[1:-1]
	wts = np.zeros((n))
	x = sympy.var('x')
	for i in range(n):
		li = lagrange_poly_support(i,pts,x)
		wts[i] = sympy.integrate(li,(x,-1.,1.))
		
	return pts,wts

#NOTE: THIS RULE IS INTENDED ONLY FOR PLOTTING, AND THEREFORE WTS IS USELESS
def Pascal1D(n):
	pts = np.linspace(-1.,1.,2*n+1)
	pts = pts[1:-1:2]
	wts = np.zeros((n))
	return pts,wts
	
def GaussLegendre1D(n):

    pts = np.array(gauss_legendre(n), dtype=np.float64)
    x = sympy.var('x')
    p = sympy.polys.orthopolys.legendre_poly(n, x)  # legendre_poly(n)
    wts = np.zeros((n))
    pd = sympy.diff(p, x)
    pdfunc = sympy.lambdify(x, pd, "numpy")
    diffp = pdfunc(pts)
    wts = 2./((1.-np.square(pts))*np.square(diffp))

    return pts, wts

# This will fail for numerical assembly, since it returns a list instead of a Numpy array


def GaussLegendre1DExact(n):
    if n == 1:
        pts = [0, ]
        wts = [2, ]
    if n == 2:
        pts = [-1/sympy.sqrt(3), 1/sympy.sqrt(3)]
        wts = [1, 1]
    if n == 3:
        pts = [-sympy.sqrt(sympy.Rational(3, 5)), 0, sympy.sqrt(sympy.Rational(3, 5))]
        wts = [sympy.Rational(5, 9), sympy.Rational(8, 9), sympy.Rational(5, 9)]
    if n == 4:
        ap = sympy.sqrt(sympy.Rational(3, 7) - sympy.Rational(2, 7)*sympy.sqrt(sympy.Rational(6, 5)))
        bp = sympy.sqrt(sympy.Rational(3, 7) + sympy.Rational(2, 7)*sympy.sqrt(sympy.Rational(6, 5)))
        aw = sympy.Rational(18, 36) + sympy.sqrt(sympy.Rational(30, 36*36))
        bw = sympy.Rational(18, 36) - sympy.sqrt(sympy.Rational(30, 36*36))
        pts = [-bp, -ap, ap, bp]
        wts = [bw, aw, aw, bw]

    if n == 5:
        pts = []
        wts = []

    return pts, wts

# This will fail for numerical assembly, since it returns a list instead of a Numpy array


def GaussLobattoLegendre1DExact(n):
    if n == 1:  # THIS IS USEFUL FOR DISPERSION STUFF (FOR GETTING SPTS CORRECT FOR DG0 SPACES) DESPITE NOT ACTUALLY EXISTING
        pts = [0, ]
        wts = [2, ]
    if n == 2:
        pts = [-1, 1]
        wts = [1, 1]
    if n == 3:
        pts = [-1, 0, 1]
        wts = [sympy.Rational(1, 3), sympy.Rational(4, 3), sympy.Rational(1, 3)]
    if n == 4:
        pts = [-1, -1/sympy.sqrt(5), 1/sympy.sqrt(5), 1]
        wts = [sympy.Rational(1, 6), sympy.Rational(5, 6), sympy.Rational(5, 6), sympy.Rational(1, 6)]
    if n == 5:
        ap = sympy.sqrt(sympy.Rational(3, 7))
        aw = sympy.Rational(49, 90)
        pts = [-1, -ap, 0, ap, 1]
        wts = [sympy.Rational(1, 10), aw, sympy.Rational(32, 45), aw, sympy.Rational(1, 10)]

    if n == 6:

        ap = sympy.sqrt(sympy.Rational(1, 3) - 2*sympy.sqrt(sympy.Rational(7, 21*21)))
        bp = sympy.sqrt(sympy.Rational(1, 3) + 2*sympy.sqrt(sympy.Rational(7, 21*21)))
        aw = sympy.Rational(14, 30) + sympy.sqrt(sympy.Rational(7, 30*30))
        bw = sympy.Rational(14, 30) - sympy.sqrt(sympy.Rational(7, 30*30))
        pts = [-1, -bp, -ap, ap, bp, 1]
        wts = [sympy.Rational(1, 15), bw, aw, aw, bw, sympy.Rational(1, 15)]

    if n == 7:
        ap = sympy.sqrt(sympy.Rational(5, 11) - 2*sympy.sqrt(sympy.Rational(5, 3*11*11)))
        bp = sympy.sqrt(sympy.Rational(5, 11) + 2*sympy.sqrt(sympy.Rational(5, 3*11*11)))
        aw = sympy.Rational(124, 350) + 7*sympy.sqrt(sympy.Rational(15, 350*350))
        bw = sympy.Rational(124, 350) - 7*sympy.sqrt(sympy.Rational(15, 350*350))
        pts = [-1, -bp, -ap, 0, ap, bp, 1]
        wts = [sympy.Rational(1, 21), bw, aw, sympy.Rational(256, 525), aw, bw, sympy.Rational(1, 21)]

    return pts, wts


def GaussLobattoLegendre1D(n):
    if n == 1:  # THIS IS USEFUL FOR DISPERSION STUFF (FOR GETTING SPTS CORRECT FOR DG0 SPACES) DESPITE NOT ACTUALLY EXISTING
        pts = np.array([0., ], dtype=np.float64)
        wts = np.array([2., ], dtype=np.float64)
    if n == 2:
        pts = np.array([-1., 1.], dtype=np.float64)
        wts = np.array([1., 1.], dtype=np.float64)
    if n >= 3:
        pts = np.array(gauss_lobatto(n), dtype=np.float64)
        wts = np.zeros((n))
        x = sympy.var('x')
        p = sympy.polys.orthopolys.legendre_poly(n-1, x)  # legendre_poly(n)
        interior_pts = pts[1:-1]
        p = sympy.polys.orthopolys.legendre_poly(n-1, x)  # legendre_poly(n)
        pfunc = sympy.lambdify(x, p, "numpy")
        pnum = pfunc(interior_pts)
        wts[1:-1] = 2./((n*(n-1.))*np.square(pnum))
        wts[0] = 2./(n*(n-1.))
        wts[-1] = 2./(n*(n-1.))
    return pts, wts
