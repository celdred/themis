import numpy as np
import sympy
import mpmath

# From PyWENO #

################################################################################
# polynomial generator, roots etc


def lagrange_poly(n, p, var):
    '''Return Langrange polynomial P_n(x) for the pth Gauss-Lobato point.
    :param n: polynomial degree
    :param p: Gauss-Lobato point
    '''
    glpts = gauss_lobatto(n+1)
    ys = np.zeros((n+1))
    ys[p] = 1.0
    return sympy.polys.specialpolys.interpolating_poly(len(glpts), var, X=glpts, Y=ys)


def legendre_poly(n):
    '''Return Legendre polynomial P_n(x).
    :param n: polynomial degree
    '''

    x = sympy.var('x')
    p = (1.0*x**2 - 1.0)**n
    top = p.diff(x, n)
    bot = 2**n * 1.0*sympy.factorial(n)
    return (top / bot).as_poly()


def find_roots(p):
    '''Return set of roots of polynomial *p*.
    :param p: sympy polynomial
    This uses the *nroots* method of the SymPy polynomial class to give
    rough roots, and subsequently refines these roots to arbitrary
    precision using mpmath.
    Returns a sorted *set* of roots.
    '''

    x = sympy.var('x')
    roots = set()

    for x0 in p.nroots():
        xi = mpmath.findroot(lambda z: p.eval(x, z), x0)
        roots.add(xi)

    return sorted(roots)


################################################################################
# quadrature points


def gauss_legendre(n):
    '''Return Gauss-Legendre nodes.
    Gauss-Legendre nodes are roots of P_n(x).
    '''

    p = legendre_poly(n)
    r = find_roots(p)
    return r


def gauss_lobatto(n):
    """Return Gauss-Lobatto nodes.
    Gauss-Lobatto nodes are roots of P'_{n-1}(x) and -1, 1.
    """

    x = sympy.var('x')
    p = legendre_poly(n-1).diff(x)
    r = find_roots(p)
    r = [mpmath.mpf('-1.0'), mpmath.mpf('1.0')] + r
    return sorted(r)


def gauss_radau(n):
    '''Return Gauss-Radau nodes.
    Gauss-Radau nodes are roots of P_n(x) + P_{n-1}(x).
    '''

    p = legendre_poly(n) + legendre_poly(n-1)
    r = find_roots(p)
    return r

##########################################
# My own stuff


def LagrangePoly(x, order, i, xi=None):
    if not (xi is not None):
        xi = sympy.symbols('x:%d' % (order+1))
    index = list(range(order+1))
    index.pop(i)
    return sympy.prod([(x-xi[j])/(xi[i]-xi[j]) for j in index])


def lagrange_poly_support(i, p, x):
    '''Give a set of N support points, an index i, and a variable x, return Langrange polynomial P_i(x)
    which interpolates the values (0,0,..1,..,0) where 1 is at the ith support point
    :param i: non-zero location
    :param p: set of N support points
    :param x: symbolic variable
    '''
    try:
        n = len(p)
    except TypeError:
        n = p.shape[0]
    return LagrangePoly(x, n-1, i, xi=p)
