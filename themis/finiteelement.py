import numpy as np
from ufl import VectorElement, TensorProductElement, EnrichedElement, HDivElement, HCurlElement, FiniteElement
import sympy
from lagrange import lagrange_poly_support  # gauss_lobatto,
from quadrature import ThemisQuadratureNumerical

# from petscshim import PETSc

variant_to_elemname = {}
variant_to_elemname['feecH1'] = 'CG'
variant_to_elemname['feecL2'] = 'DG'
variant_to_elemname['mseH1'] = 'CMSE'
variant_to_elemname['mseL2'] = 'DMSE'
variant_to_elemname['qbH1'] = 'CQB'
variant_to_elemname['qbL2'] = 'DQB'
variant_to_elemname['mgdH1'] = 'GD'
variant_to_elemname['mgdL2'] = 'DGD'



def extract_element_info(elem):
    degree = elem.degree()
    variant = elem.variant()
    if variant is None:
        variant = 'feec'

    degreelist = []
    elemnamelist = []
    variantlist = []
    contlist = []

    if elem.cell().cellname() == 'interval':
        ncomp = 1
        degreelist.append([degree, ])
        variantlist.append([variant, ])
        if elem.family() == 'Discontinuous Lagrange':
            elemnamelist.append([variant_to_elemname[variant + 'L2'], ])
            contlist.append(['L2', ])
            dofmap = (0,degree+1)
        elif elem.family() == 'Lagrange':
            elemnamelist.append([variant_to_elemname[variant + 'H1'], ])
            contlist.append(['H1', ])
            dofmap = (1,degree-1)
        else:
            raise ValueError('themis supports only CG/DG on intervals')

    elif elem.cell().cellname() == 'quadrilateral':
        if elem.family() in ['DQ', 'Q']:
            variantlist.append([variant, variant])
            degreelist.append([degree, degree])
            if elem.family() == 'DQ':
                elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2']])
                contlist.append(['L2', 'L2', ])
                dofmap = (0,0,(degree+1)*(degree+1))
            if elem.family() == 'Q':
                elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1']])
                contlist.append(['H1', 'H1', ])
                dofmap = ()
            ncomp = 1
        elif elem.family() == 'RTCF':
            variantlist.append([variant, variant])
            variantlist.append([variant, variant])
            degreelist.append([degree, degree-1])
            degreelist.append([degree-1, degree])
            contlist.append(['H1', 'L2'])
            contlist.append(['L2', 'H1'])
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
            ncomp = 2
        elif elem.family() == 'RTCE':
            variantlist.append([variant, variant])
            variantlist.append([variant, variant])
            degreelist.append([degree-1, degree])
            degreelist.append([degree, degree-1])
            contlist.append(['L2', 'H1'])
            contlist.append(['H1', 'L2'])
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
            ncomp = 2
        else:
            raise ValueError('themis supports only Q/DQ/RTCF/RTCE on quads')

    elif elem.cell().cellname() == 'hexahedron':
        if elem.family() in ['DQ', 'Q']:
            variantlist.append([variant, variant, variant])
            degreelist.append([degree, degree, degree])
            if elem.family() == 'DQ':
                elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2']])
                contlist.append(['L2', 'L2', 'L2'])
            if elem.family() == 'Q':
                elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1']])
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
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2']])
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
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
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1']])
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
            ncomp = 3
        else:
            raise ValueError('themis supports only Q/DQ/NCF/NCE on hexahedrons')
    else:
        raise ValueError('Themis does not support cell type %s',elem.cell())

    return ncomp, variantlist, degreelist, elemnamelist, contlist, dofmap

# ADD CORRECT DOFMAP STUFF
def flatten_tp_element(elem1, elem2):
    ncomp1, variantlist1, degreelist1, elemnamelist1, contlist1, dofmap1 = extract_element_info(elem1)
    ncomp2, variantlist2, degreelist2, elemnamelist2, contlist2, dofmap2 = extract_element_info(elem2)
    dofmap = dofmap1 + dofmap2 # THIS IS WRONG!
    for c1 in range(ncomp1):
        variantlist1[c1].append(variantlist2[0][0])
        degreelist1[c1].append(degreelist2[0][0])
        elemnamelist1[c1].append(elemnamelist2[0][0])
        contlist1[c1].append(contlist2[0][0])
    return ncomp1, variantlist1, degreelist1, elemnamelist1, contlist1, dofmap

# ADD CORRECT DOFMAP STUFF
def merge_enriched_element(elem1, elem2):
    if isinstance(elem1, FiniteElement):
        ncomp1, variantlist1, degreelist1, elemnamelist1, contlist1, dofmap1 = extract_element_info(elem1)
    elif isinstance(elem1, TensorProductElement):
        subelem1, subelem2 = elem1.sub_elements()
        ncomp1, variantlist1, degreelist1, elemnamelist1, contlist1, dofmap1 = flatten_tp_element(subelem1, subelem2)

    if isinstance(elem2, FiniteElement):
        ncomp2, variantlist2, degreelist2, elemnamelist2, contlist2, dofmap2 = extract_element_info(elem2)
    elif isinstance(elem2, TensorProductElement):
        subelem1, subelem2 = elem2.sub_elements()
        ncomp2, variantlist2, degreelist2, elemnamelist2, contlist2, dofmap2 = flatten_tp_element(subelem1, subelem2)
    
    dofmap = dofmap1 + dofmap2 # THIS IS WRONG!
    ncomp = ncomp1 + ncomp2
    variantlist = variantlist1 + variantlist2
    degreelist = degreelist1 + degreelist2
    elemnamelist = elemnamelist1 + elemnamelist2
    contlist = contlist1 + contlist2
    return ncomp, variantlist, degreelist, elemnamelist, contlist, dofmap


class ThemisElement():
    def __init__(self, elem, sptsh1=None, sptsl2=None):

        # This all maybe becomes an extract element info function?
        
        # ADD SUPPORT FOR TENSOR ELEMENTS
        if isinstance(elem, VectorElement):
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
            ncomp, variantlist, degreelist, elemnamelist, contlist = merge_enriched_element(elem1, elem2)
        elif isinstance(elem, TensorProductElement):
            elem1, elem2 = elem.sub_elements()
            if not (elem2.cell().cellname() == 'interval'):
                raise ValueError('Themis only supports tensor product elements with the second element on an interval')
            if not (isinstance(elem1, FiniteElement) and isinstance(elem2, FiniteElement)):
                raise ValueError('Themis supports only tensor product elements of FiniteElement')
            ncomp, variantlist, degreelist, elemnamelist, contlist = flatten_tp_element(elem1, elem2)
        elif isinstance(elem, FiniteElement):
            ncomp, variantlist, degreelist, elemnamelist, contlist = extract_element_info(elem)
        else:
            raise ValueError('Themis supports only FiniteElemet, EnrichedElement and TensorProductElement')
    
# CHECK THIS...
        self._cont = elem.sobolev_space()
# THIS NEEDS TO DETERMINE IF BASIS IS INTERPOLATORY AS WELL!
        self._interpolatory = False
# BASICALLY ONLY IF ALL COMPONENTS AND SUBELEMENTS ARE INTERPOLATORY AS WELL!

        self._ncomp = ncomp
        self._elemnamelist = elemnamelist
        self._degreelist = degreelist
        self._contlist = contlist
        self._variantlist = variantlist

        self._nbasis = []
        self._nblocks = []
        self._basis = []
        self._derivs = []
        self._derivs2 = []
        self._spts = []

# FIX THIS
        self._location =
        self._dofnums =
        self._cell =

        for ci in range(self._ncomp):

            self._nbasis.append([])
            self._nblocks.append([])
            self._basis.append([])
            self._derivs.append([])
            self._derivs2.append([])
            self._spts.append([])
            # ADD LOCATION
            # ADD DOFNUMS
            # ADD CELLX/CELLY/ETC.
            for elemname, degree, variant in zip(self._elemnamelist[ci], self._degreelist[ci], self._variantlist[ci]):

                sptsL2 = sptsl2 or ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]  # this works because degree has already been adjusted!
                sptsH1 = sptsh1 or ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]
                
                # compute number of shape functions in each direction (=nbasis)
                self._nbasis[ci].append(degree + 1)
 
                # set the number of blocks
                if elemname in ['CG', 'CQB', 'CMSE', 'DG', 'DQB', 'DMSE']:
                    self._nblocks[ci].append(1)
                if elemname == 'GD':
                    self._nblocks[ci].append(degree)
                if elemname == 'DGD':
                    self._nblocks[ci].append(degree+1)

                # compute basis and deriv functions
                if elemname == 'CG':
                    b, d, d2, s = _CG_basis(degree, sptsH1)
                if elemname == 'DG':
                    b, d, d2, s = _DG_basis(degree, sptsL2)
                if elemname == 'GD':
                    b, d, d2, s = _GD_basis(degree)
                if elemname == 'DGD':
                    b, d, d2, s = _DGD_basis(degree)
                if elemname == 'CQB':
                    b, d, d2, s = _CQB_basis(degree)
                if elemname == 'DQB':
                    b, d, d2, s = _DQB_basis(degree)
                if elemname == 'CMSE':
                    b, d, d2, s = _CMSE_basis(degree, sptsH1)
                if elemname == 'DMSE':
                    b, d, d2, s = _DMSE_basis(degree, sptsH1)  # Note use of sptsH1 here, since DMSE basis is built from CMSE basis!
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)

            # add EmptyElements in
            while len(self._basis[ci]) < 3:
                b, d, d2, s = _DG_basis(0, [0.5, ])
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)
                self._nblocks[ci].append(1)

    def get_continuity(self, ci, direc):
        return self._contlist[ci][direc]
    
    def dofmap(self):
        return FIXME
        
    def ndofs(self):
        return self._ndofs

    def get_nbasis(self, ci, direc):
        return self._nbasis[ci][direc]

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

# THIS IS NOW WRONG FOR FEEC TYPE ELEMENTS SINCE WE ARE NOT LYING ABOUT ELEMENTS ANYMORE
    def swidth(self):
        maxnbasis = max(max(self._nbasis))
        return maxnbasis

    def get_ncomp(self):
        return self._ncomp
    
# THIS IS REALLY A CONSTANT, EXCEPT FOR TP ELEMENTS...
    def get_nblocks(self, ci, direc):
        return self._nblocks[ci][direc]


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
    if order >= 1:
        cg_symbas, _, _, _ = _CQB_basis(order+1, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2, None


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
