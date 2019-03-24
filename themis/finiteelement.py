import numpy as np
from ufl import VectorElement, TensorProductElement, EnrichedElement, HDivElement, HCurlElement, FiniteElement, TensorElement
import sympy
from lagrange import lagrange_poly_support, gauss_lobatto

def GaussLobattoLegendre1D(n):
    if n == 1:  # THIS IS USEFUL FOR DISPERSION STUFF (FOR GETTING SPTS CORRECT FOR DG0 SPACES) DESPITE NOT ACTUALLY EXISTING
        pts = np.array([0., ], dtype=np.float64)
    if n == 2:
        pts = np.array([-1., 1.], dtype=np.float64)
    if n >= 3:
        pts = np.array(gauss_lobatto(n), dtype=np.float64)
        x = sympy.var('x')
        p = sympy.polys.orthopolys.legendre_poly(n-1, x)  # legendre_poly(n)
        interior_pts = pts[1:-1]
        p = sympy.polys.orthopolys.legendre_poly(n-1, x)  # legendre_poly(n)
        pfunc = sympy.lambdify(x, p, "numpy")
        pnum = pfunc(interior_pts)
    return 0.5 * pts + 0.5 # scale pts to [0,1]

def extract_element_info(elem):
    degree = elem.degree()
    variant = elem.variant()
    if variant is None:
        raise ValueError("Themis only supports elements with variant set")
    if not variant in ['mgd', 'feec', 'mse', 'qb']:
        raise ValueError('Themis doesnt know how to handle variant %s', variant)

    degreelist = []
    variantlist = []
    contlist = []

    if elem.cell().cellname() == 'interval':
        ncomp = 1
        degreelist.append([degree, ])
        variantlist.append([variant, ])
        if elem.family() == 'Discontinuous Lagrange':
            contlist.append(['L2', ])
        elif elem.family() == 'Lagrange':
            contlist.append(['H1', ])
        else:
            raise ValueError('themis supports only CG/DG on intervals')

    elif elem.cell().cellname() == 'quadrilateral':
        if elem.family() in ['DQ', 'Q']:
            variantlist.append([variant, variant])
            degreelist.append([degree, degree])
            if elem.family() == 'DQ':
                contlist.append(['L2', 'L2', ])
            if elem.family() == 'Q':
                contlist.append(['H1', 'H1', ])
            ncomp = 1
        elif elem.family() == 'RTCF':
            variantlist.append([variant, variant])
            variantlist.append([variant, variant])
            degreelist.append([degree, degree-1])
            degreelist.append([degree-1, degree])
            contlist.append(['H1', 'L2'])
            contlist.append(['L2', 'H1'])
            ncomp = 2
        elif elem.family() == 'RTCE':
            variantlist.append([variant, variant])
            variantlist.append([variant, variant])
            degreelist.append([degree-1, degree])
            degreelist.append([degree, degree-1])
            contlist.append(['L2', 'H1'])
            contlist.append(['H1', 'L2'])
            ncomp = 2
        else:
            raise ValueError('themis supports only Q/DQ/RTCF/RTCE on quads')

    elif elem.cell().cellname() == 'hexahedron':
        if elem.family() in ['DQ', 'Q']:
            variantlist.append([variant, variant, variant])
            degreelist.append([degree, degree, degree])
            if elem.family() == 'DQ':
                contlist.append(['L2', 'L2', 'L2'])
            if elem.family() == 'Q':
                contlist.append(['H1', 'H1', 'H1'])
            ncomp = 1
        elif elem.family() == 'NCF':
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            degreelist.append([degree, degree-1, degree-1])
            degreelist.append([degree-1, degree, degree-1])
            degreelist.append([degree-1, degree-1, degree])
            contlist.append(['H1', 'L2', 'L2'])
            contlist.append(['L2', 'H1', 'L2'])
            contlist.append(['L2', 'L2', 'H1'])
            ncomp = 3
        elif elem.family() == 'NCE':
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            degreelist.append([degree-1, degree, degree])
            degreelist.append([degree, degree-1, degree])
            degreelist.append([degree, degree, degree-1])
            contlist.append(['L2', 'H1', 'H1'])
            contlist.append(['H1', 'L2', 'H1'])
            contlist.append(['H1', 'H1', 'L2'])
            ncomp = 3
        else:
            raise ValueError('themis supports only Q/DQ/NCF/NCE on hexahedrons')
    else:
        raise ValueError('Themis does not support this cell type')

    return ncomp, variantlist, degreelist, contlist


def flatten_tp_element(elem1, elem2):
    ncomp1, variantlist1, degreelist1, contlist1 = extract_element_info(elem1)
    ncomp2, variantlist2, degreelist2, contlist2 = extract_element_info(elem2)
    for c1 in range(ncomp1):
        variantlist1[c1].append(variantlist2[0][0])
        degreelist1[c1].append(degreelist2[0][0])
        contlist1[c1].append(contlist2[0][0])
    return ncomp1, variantlist1, degreelist1, contlist1


def merge_enriched_element(elem1, elem2):
    if isinstance(elem1, FiniteElement):
        ncomp1, variantlist1, degreelist1, contlist1 = extract_element_info(elem1)
    elif isinstance(elem1, TensorProductElement):
        subelem1, subelem2 = elem1.sub_elements()
        ncomp1, variantlist1, degreelist1, contlist1 = flatten_tp_element(subelem1, subelem2)

    if isinstance(elem2, FiniteElement):
        ncomp2, variantlist2, degreelist2, contlist2 = extract_element_info(elem2)
    elif isinstance(elem2, TensorProductElement):
        subelem1, subelem2 = elem2.sub_elements()
        ncomp2, variantlist2, degreelist2, contlist2 = flatten_tp_element(subelem1, subelem2)

    ncomp = ncomp1 + ncomp2
    variantlist = variantlist1 + variantlist2
    degreelist = degreelist1 + degreelist2
    contlist = contlist1 + contlist2
    return ncomp, variantlist, degreelist, contlist


class ThemisElement():
    # THESE SPTS SHOULD REALLY BE DIFFERENT VARIANTS I THINK...
    # THIS IS USEFUL MOSTLY FOR DISPERSION STUFF
    # BUT I THINK REALLY IMPLEMENTING A BUNCH OF VARIANTS IS THE BEST APPROACH
    def __init__(self, elem, sptsh1=None, sptsl2=None):

        # ADD SUPPORT FOR TENSOR ELEMENTS
        if isinstance(elem, TensorElement):
            self._ndofs = elem.value_size() # IS THIS CORRECT?
# CHECK THIS
        elif isinstance(elem, VectorElement):
            self._ndofs = elem.value_size()  # do this BEFORE extracting baseelem
            elem = elem.sub_elements()[0]
        else:
            self._ndofs = 1

        if isinstance(elem, EnrichedElement):
            elem1, elem2 = elem._elements
            if (isinstance(elem1, HDivElement) and isinstance(elem2, HDivElement)) or (isinstance(elem1, HCurlElement) and isinstance(elem2, HCurlElement)):
                elem1 = elem1._element
                elem2 = elem2._element
            else:
                raise ValueError('Themis supports only EnrichedElement made of 2 HDiv/HCurl elements')
            if not ((isinstance(elem1, FiniteElement) or isinstance(elem1, TensorProductElement)) and (isinstance(elem2, FiniteElement) or isinstance(elem2, TensorProductElement))):
                raise ValueError('Themis supports only EnrichedElement made of 2 HDiv/HCurl elements that are themselves FiniteElement or TensorProductElement')
            ncomp, variantlist, degreelist, contlist = merge_enriched_element(elem1, elem2)
        elif isinstance(elem, TensorProductElement):
            elem1, elem2 = elem.sub_elements()
            if not (elem2.cell().cellname() == 'interval'):
                raise ValueError('Themis only supports tensor product elements with the second element on an interval')
            if not (isinstance(elem1, FiniteElement) and isinstance(elem2, FiniteElement)):
                raise ValueError('Themis supports only tensor product elements of FiniteElement')
            ncomp, variantlist, degreelist, contlist = flatten_tp_element(elem1, elem2)
        elif isinstance(elem, FiniteElement):
            ncomp, variantlist, degreelist, contlist = extract_element_info(elem)
        else:
            raise ValueError('Themis supports only FiniteElemet, EnrichedElement and TensorProductElement')

        self._cont = elem.sobolev_space()


        self._ncomp = ncomp
        self._degreelist = degreelist
        self._contlist = contlist
        self._variantlist = variantlist

        self._nbasis = []
        self._nblocks = []
        self._ndofs_per_element = []
        self._offsets = []
        self._offset_mult = []
        self._basis = []
        self._derivs = []
        self._derivs2 = []
        self._spts = []
        self._entries_of = []
        self._entries_om = []

        self.interpolatory = True
        if self._ncomp > 1:
            self.interpolatory = False

        for ci in range(self._ncomp):

            self._nbasis.append([])
            self._nblocks.append([])
            self._ndofs_per_element.append([])
            self._offsets.append([])
            self._offset_mult.append([])
            self._basis.append([])
            self._derivs.append([])
            self._derivs2.append([])
            self._spts.append([])
            self._entries_of.append([])
            self._entries_om.append([])
            for degree, variant, cont in zip(self._degreelist[ci], self._variantlist[ci], self._contlist[ci]):

                if not ((variant in ['mse', 'feec', 'mgd'] and cont == 'H1') or (variant == 'feec' and cont == 'L2')):
                    self.interpolatory = False

                if variant == 'mse' and cont == 'L2':  # this accounts for the fact that DMSE is built using CMSE basis from complex, which counts from 1!
                    sptsL2 = sptsl2 or GaussLobattoLegendre1D(degree+2)
                else:
                    sptsL2 = sptsl2 or GaussLobattoLegendre1D(degree+1)  # count starts at 0
                sptsH1 = sptsh1 or GaussLobattoLegendre1D(degree+1)

                # compute number of shape functions in each direction (=nbasis); and number of degrees of freedom per element
                self._nbasis[ci].append(degree + 1)
                if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
                    self._ndofs_per_element[ci].append(degree)
                if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
                    self._ndofs_per_element[ci].append(degree + 1)
                if variant == 'mgd':
                    self._ndofs_per_element[ci].append(1)

                # set the number of blocks
                if variant in ['feec', 'qb', 'mse']:
                    self._nblocks[ci].append(1)
                if variant == 'mgd' and cont == 'H1':
                    self._nblocks[ci].append(degree)
                if variant == 'mgd' and cont == 'L2':
                    self._nblocks[ci].append(degree+1)

                # compute offsets and offset mults
                if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
                    of, om = _CG_offset_info(degree)
                if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
                    of, om = _DG_offset_info(degree)
                if variant == 'mgd' and cont == 'H1':
                    of, om = _GD_offset_info(degree)
                if variant == 'mgd' and cont == 'L2':
                    of, om = _DGD_offset_info(degree)
                self._offsets[ci].append(of)
                self._offset_mult[ci].append(om)

                # compute entries
                if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
                    of, om = _CG_entries_info(degree)
                if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
                    of, om = _DG_entries_info(degree)
                if variant == 'mgd' and cont == 'H1':
                    of, om = _GD_entries_info(degree)
                if variant == 'mgd' and cont == 'L2':
                    of, om = _DGD_entries_info(degree)
                self._entries_of[ci].append(of)
                self._entries_om[ci].append(om)

                # compute basis and deriv functions
                if variant == 'feec' and cont == 'H1':
                    b, d, d2, s = _CG_basis(degree, sptsH1)
                if variant == 'feec' and cont == 'L2':
                    b, d, d2, s = _DG_basis(degree, sptsL2)
                if variant == 'mgd' and cont == 'H1':
                    b, d, d2, s = _GD_basis(degree)
                if variant == 'mgd' and cont == 'L2':
                    b, d, d2, s = _DGD_basis(degree)
                if variant == 'qb' and cont == 'H1':
                    b, d, d2, s = _CQB_basis(degree)
                if variant == 'qb' and cont == 'L2':
                    b, d, d2, s = _DQB_basis(degree)
                if variant == 'mse' and cont == 'H1':
                    b, d, d2, s = _CMSE_basis(degree, sptsH1)
                if variant == 'mse' and cont == 'L2':
                    b, d, d2, s = _DMSE_basis(degree, sptsL2)  # Note that sptsL2 here is actually the H1 spts for degree+1
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)

            # add EmptyElements in
            while len(self._basis[ci]) < 3:
                of, om = _DG_offset_info(0)
                ofe, ome = _DG_entries_info(0)
                b, d, d2, s = _DG_basis(0, [0.5, ])
                self._offsets[ci].append(of)
                self._offset_mult[ci].append(om)
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)
                self._nblocks[ci].append(1)
                self._entries_of[ci].append(ofe)
                self._entries_om[ci].append(ome)
                self._contlist[ci].append('L2')

    def get_continuity(self, ci, direc):
        return self._contlist[ci][direc]

    def get_nx(self, ci, direc, ncell, bc):
        variant = self._variantlist[ci][direc]
        cont = self._contlist[ci][direc]
        if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
            nx = self._degreelist[ci][direc] * ncell
            if (bc == 'nonperiodic'):
                nx = nx + 1
        if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
            nx = (self._degreelist[ci][direc]+1) * ncell
        if variant == 'mgd' and cont == 'H1':
            nx = ncell
            if (bc == 'nonperiodic'):
                nx = nx + 1
        if variant == 'mgd' and cont == 'L2':
            nx = ncell
        return nx

    def ndofs(self):
        return self._ndofs

    def get_ndofs_per_element(self, ci, direc):
        return self._ndofs_per_element[ci][direc]

    def get_entries(self, ci, direc):
        return self._entries_of[ci][direc], self._entries_om[ci][direc]

# Offsets are returned as nblock-nbasis
# Basis/Derivs are returned as nblock-nquad-nbasis (tabulated) or nblock-nbasis (symbolic)

    def get_offsets(self, ci, direc):
        return self._offsets[ci][direc], self._offset_mult[ci][direc]

    def get_sym_basis(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        blist = []
        for bi in range(self._nblocks[ci][direc]):
            blist.append([])
            for basis in self._basis[ci][direc][bi]:
                adjbasis = basis.subs(xsymb, newsymb)
                blist[bi].append(adjbasis)
        return blist

    def get_sym_derivs(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for bi in range(self._nblocks[ci][direc]):
            dlist.append([])
            for deriv in self._derivs[ci][direc][bi]:
                adjderiv = deriv.subs(xsymb, newsymb)
                dlist[bi].append(adjderiv)
        return dlist

    def get_sym_derivs2(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for bi in range(self._nblocks[ci][direc]):
            dlist.append([])
            for deriv2 in self._derivs2[ci][direc][bi]:
                adjderiv2 = deriv2.subs(xsymb, newsymb)
                dlist[bi].append(adjderiv2)
        return dlist

    def get_basis_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = len(x)
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_funcs = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self._basis[ci][direc][bi]
            for i in range(nbasis):
                for j in range(npts):
                    if symbas[i] == 1.0:  # check for the constant basis function
                        basis_funcs[bi, j, i] = sympy.Rational(1, 1)
                else:
                    basis_funcs[bi, j, i] = symbas[i].subs(xsymb, x[j])
        return basis_funcs

    def get_derivs_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = len(x)
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_derivs = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self._derivs[ci][direc][bi]
            for i in range(nbasis):
                for j in range(npts):
                    if symbas[i] == 1.0:  # check for the constant basis function
                        basis_derivs[bi, j, i] = sympy.Rational(1, 1) * 0
                else:
                    basis_derivs[bi, j, i] = symbas[i].subs(xsymb, x[j])
        return basis_derivs

    def get_derivs2_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = len(x)
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_derivs2 = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self._derivs2[ci][direc][bi]
            for i in range(nbasis):
                for j in range(npts):
                    if symbas[i] == 1.0:  # check for the constant basis function
                        basis_derivs2[bi, j, i] = sympy.Rational(1, 1) * 0
                else:
                    basis_derivs2[bi, j, i] = symbas[i].subs(xsymb, x[j])
        return basis_derivs2

    def get_basis(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = x.shape[0]
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_funcs = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self._basis[ci][direc][bi]
            for i in range(len(symbas)):
                if symbas[i] == 1.0:  # check for the constant basis function
                    basis_funcs[bi, :, i] = 1.0
                else:
                    basis = sympy.lambdify(xsymb, symbas[i], "numpy")
                    basis_funcs[bi, :, i] = np.squeeze(basis(x))
        return basis_funcs

    def get_derivs(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = x.shape[0]
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_derivs = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self._derivs[ci][direc][bi]
            for i in range(len(symbas)):
                if symbas[i] == 0.0:  # check for the constant basis function
                    basis_derivs[bi, :, i] = 0.0
                else:
                    deriv = sympy.lambdify(xsymb, symbas[i], "numpy")
                    basis_derivs[bi, :, i] = np.squeeze(deriv(x))
        return basis_derivs

    def get_derivs2(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = x.shape[0]
        nbasis = len(self._basis[ci][direc][0])
        nblocks = self._nblocks[ci][direc]
        basis_derivs2 = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self._derivs2[ci][direc][bi]
            for i in range(len(symbas)):
                if symbas[i] == 0.0:  # check for the constant basis function
                    basis_derivs2[bi, :, i] = 0.0
                else:
                    deriv2 = sympy.lambdify(xsymb, symbas[i], "numpy")
                    basis_derivs2[bi, :, i] = np.squeeze(deriv2(x))
        return basis_derivs2

    def swidth(self):
        maxnbasis = max(max(self._nbasis))
        return maxnbasis

    def get_ncomp(self):
        return self._ncomp

    def get_nblocks(self, ci, direc):
        return self._nblocks[ci][direc]

    def get_spts(self, ci, direc):
        return self._spts[ci][direc]


# Lagrange Elements

def _CG_basis(order, spts, symb=None):
    xsymb = symb or sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    for i in range(order+1):
        symbas.append(lagrange_poly_support(i, spts, xsymb))
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    return [symbas, ], [derivs, ], [derivs2, ], spts


def _DG_basis(order, spts, symb=None):
    xsymb = symb or sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    if order >= 1:
        for i in range(order+1):
            symbas.append(lagrange_poly_support(i, spts, xsymb))
            derivs.append(sympy.diff(symbas[i]))
            derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    else:
        spts = [sympy.Rational(1, 2), ]
        symbas.append(sympy.Rational(1, 1))
        derivs.append(sympy.Rational(1, 1) * 0)
        derivs2.append(sympy.Rational(1, 1) * 0)
    return [symbas, ], [derivs, ], [derivs2, ], spts


def _CG_offset_info(order):
    offsets = np.arange(0, order+1, dtype=np.int32)
    offset_multiplier = order * np.ones(offsets.shape, dtype=np.int32)
    # 'p*i','p*i+1'...'p*i+p'
    return np.expand_dims(offsets, axis=0), np.expand_dims(offset_multiplier, axis=0)


def _CG_entries_info(order):
    of, om = _CG_offset_info(order)
    return of[0], om[0]


def _DG_offset_info(order):
    offsets = np.arange(0, order+1, dtype=np.int32)
    offset_multiplier = (order+1) * np.ones(offsets.shape, dtype=np.int32)
    # 'p*i','p*i+1'...'p*i+p'
    return np.expand_dims(offsets, axis=0), np.expand_dims(offset_multiplier, axis=0)


def _DG_entries_info(order):
    of, om = _DG_offset_info(order)
    return of[0], om[0]


#  MGD ELEMENTS

# def direct_basis(x):
    # f1  =  .5*( x**3/3 + x**2/2 - x/12  - 1/8)               # 0 [0 0] 1
    # f2  = -.5*( x**3   + x**2/2 - 9/4*x - 9/8)               # 0 [0 1] 0
    # f3  = -.5*(-x**3   + x**2/2 + 9/4*x - 9/8)               # 0 [1 0] 0
    # f4  =  .5*(-x**3/3 + x**2/2 + x/12  - 1/8)               # 1 [0 0] 0
    # f1m =  .5*( (x-1)**3/3 + (x-1)**2/2 - (x-1)/12  - 1/8)   # [0 0] 0 1
    # f2m = -.5*( (x-1)**3   + (x-1)**2/2 - 9/4*(x-1) - 9/8)   # [0 0] 1 0
    # f3m = -.5*(-(x-1)**3   + (x-1)**2/2 + 9/4*(x-1) - 9/8)   # [0 1] 0 0
    # f4m =  .5*(-(x-1)**3/3 + (x-1)**2/2 + (x-1)/12  - 1/8)   # [1 0] 0 0
    # f1p =  .5*( (x+1)**3/3 + (x+1)**2/2 - (x+1)/12  - 1/8)   # 0 0 [0 1]
    # f2p = -.5*( (x+1)**3   + (x+1)**2/2 - 9/4*(x+1) - 9/8)   # 0 0 [1 0]
    # f3p = -.5*(-(x+1)**3   + (x+1)**2/2 + 9/4*(x+1) - 9/8)   # 0 1 [0 0]
    # f4p =  .5*(-(x+1)**3/3 + (x+1)**2/2 + (x+1)/12  - 1/8)   # 1 0 [0 0]
    # return [f4m,f3m,f2m,f1m], [f4,f3,f2,f1], [f4p,f3p,f2p,f1p]


def _GD_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    a = 2
    b = -order
    p = a * np.arange(order+1) + b
    p = sympy.Rational(1, 2) * p + sympy.Rational(1, 2)
    spts = [0, 1]
    for i in range(0, order+1):
        basis = lagrange_poly_support(i, p, xsymb)
        symbas.append(basis)
        derivs.append(sympy.diff(basis))
        derivs2.append(sympy.diff(sympy.diff(basis)))

    # do a linear combination of basis functions here for blocks near the boundary, based on extrapolation formulas
    # phihat0 = phi0 + 4 phi-1
    # phihat1 = phi1 - 6 phi-1
    # phihat2 = phi2 + 4 phi-1
    # phihat3 = phi3 - phi-1
    # phihatN = phiN + 4 phiN+1
    # phihatN-1 = phiN-1 - 6 phiN+1
    # phihatN-2 = phiN-2 + 4 phiN+1
    # phihatN-3 = phiN-3 - phiN+1
    if order == 3:
        # symbasleft, _, symbasright = direct_basis(xsymb - sympy.Rational(1,2))
        symbasleft = []
        symbasright = []
        # permutation patterns here take into account offsets
        symbasleft.append(symbas[1] + 4 * symbas[0])
        symbasleft.append(symbas[2] - 6 * symbas[0])
        symbasleft.append(symbas[3] + 4 * symbas[0])
        symbasleft.append(0 - 1 * symbas[0])
        symbasright.append(0 - 1 * symbas[3])
        symbasright.append(symbas[0] + 4 * symbas[3])
        symbasright.append(symbas[1] - 6 * symbas[3])
        symbasright.append(symbas[2] + 4 * symbas[3])
        derivsleft = []
        derivsright = []
        derivs2left = []
        derivs2right = []
        for i in range(len(symbas)):
            derivsleft.append(sympy.diff(symbasleft[i]))
            derivsright.append(sympy.diff(symbasright[i]))
            derivs2left.append(sympy.diff(sympy.diff(symbasleft[i])))
            derivs2right.append(sympy.diff(sympy.diff(symbasright[i])))
        return [symbasleft, symbas, symbasright], [derivsleft, derivs, derivsright], [derivs2left, derivs2, derivs2right], spts
# ADD SUPPORT HERE FOR MORE GENERAL FORMULAS...
    else:
        return [symbas]*order, [derivs]*order, [derivs2]*order, spts

# uses the formulas from Hiemstra et. al 2014 to create a compatible DG basis from a given CG basis


def create_compatible_basis(cg_symbas):
    nblocks = len(cg_symbas)
    ncg_basis = len(cg_symbas[0])
    symbas = []
    derivs = []
    derivs2 = []
    for bi in range(nblocks):
        symbas.append([])
        derivs.append([])
        derivs2.append([])
        for i in range(1, ncg_basis):
            basis = 0
            for j in range(i):
                basis = basis + sympy.diff(-cg_symbas[bi][j])
            symbas[bi].append(basis)
            derivs[bi].append(sympy.diff(basis))
            derivs2[bi].append(sympy.diff(sympy.diff(basis)))
    return symbas, derivs, derivs2


def _DGD_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    spts = [sympy.Rational(1, 2), ]
    if order >= 1:
        cg_symbas, _, _, _ = _GD_basis(order+1, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2, spts


def _GD_offset_info(order):
    offsets = np.arange(-(order-1)//2, (order-1)//2+2, 1, dtype=np.int32)
    offset_multiplier = np.ones(offsets.shape, dtype=np.int32)
    expanded_offset_multiplier = np.reshape(np.tile(offset_multiplier, order), (order, offset_multiplier.shape[0]))
    M = (order-1)//2  # nblocks = order
    expanded_offsets = np.zeros((order, offsets.shape[0]), dtype=np.int32)
    k = 0
    for i in range(M, -(M+1), -1):
        expanded_offsets[k, :] = offsets + i
        k = k + 1
    # print('GD',expanded_offsets)
    return expanded_offsets, expanded_offset_multiplier


def _GD_entries_info(order):
    return np.array([0, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32)


def _DGD_offset_info(order):
    offsets = np.arange(-(order)//2, (order)/2+1, 1, dtype=np.int32)
    offset_multiplier = np.ones(offsets.shape, dtype=np.int32)
    expanded_offset_multiplier = np.reshape(np.tile(offset_multiplier, order+1), (order+1, offset_multiplier.shape[0]))
    M = order//2  # nblocks = order + 1
    expanded_offsets = np.zeros((order+1, offsets.shape[0]), dtype=np.int32)
    k = 0
    for i in range(M, -(M+1), -1):
        expanded_offsets[k, :] = offsets + i
        k = k + 1
    # print('DGD',expanded_offsets)
    return expanded_offsets, expanded_offset_multiplier

# CORRECT FACET INTEGRAL STUFF??


def _DGD_entries_info(order):
    return np.array([0, ], dtype=np.int32), np.array([1, ], dtype=np.int32)


# QB ELEMENTS


def bernstein_polys(n, var):
    '''Returns the n+1 bernstein basis polynomials of degree n (n>=1) on [-1,1]'''
    # if n==0:
    # return [1/2.,]
    polys = []
    # b = 1
    # a = -1
    # t = sympy.var('t')
    for v in range(0, n+1):
        coeff = sympy.binomial(n, v)
        basisfunc = coeff * (1 - var)**(n-v) * var**v
        # basisfunc = basisfunc.subs(t,(var-a)/(b-a))
        polys.append(basisfunc)
    return polys

# HOW SHOULD SPTS BE HANDLED HERE?
# NOT EVEN REALLY SURE IT MAKES SENSE...


def _CQB_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    symbas = bernstein_polys(order, xsymb)
    derivs = []
    derivs2 = []
    for i in range(len(symbas)):
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    return [symbas, ], [derivs, ], [derivs2, ], None


def _DQB_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    spts = None
    if order >= 1:
        cg_symbas, _, _, _ = _CQB_basis(order+1, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2, spts


# MSE ELEMENTS
_CMSE_basis = _CG_basis

# Careful here, spts argument is actually those used for the CG space!


def _DMSE_basis(order, spts, symb=None):
    xsymb = symb or sympy.var('x')
    if order >= 1:
        cg_symbas, _, _, _ = _CMSE_basis(order+1, spts, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2, spts
