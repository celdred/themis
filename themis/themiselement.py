import numpy as np
import sympy
from themis.lagrange import lagrange_poly_support
from themis.quadrature import QuadratureNumerical
from ufl import FiniteElement, VectorElement, TensorElement, TensorProductElement, HDivElement, HCurlElement, EnrichedElement

__all__ = ["ThemisElement", "IntervalElement"]


def a_to_cinit_string(x):
    np.set_printoptions(threshold=np.prod(x.shape))
    sx = np.array2string(x, separator=',', precision=100)
    sx = sx.replace('\n', '')
    sx = sx.replace('[', '{')
    sx = sx.replace(']', '}')
    return sx


def extract_element_info(elem):
    degree = elem.degree()
    variant = elem.variant()
    if variant is None:
        raise ValueError("Themis only supports elements with variant set")
    if variant not in ['mgd', 'feec', 'mse', 'qb']:
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
        # This matches the construction in https://github.com/firedrakeproject/ufl/blob/master/ufl/finiteelement/finiteelement.py
        # Note that on a reference element it gives u,v for RTCF and v,u for RTCE
        elif elem.family() in ['RTCF', 'RTCE']:
            variantlist.append([variant, variant])
            variantlist.append([variant, variant])
            degreelist.append([degree, degree-1])
            degreelist.append([degree-1, degree])
            contlist.append(['H1', 'L2'])
            contlist.append(['L2', 'H1'])
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
        # This matches the construction in https://github.com/firedrakeproject/ufl/blob/master/ufl/finiteelement/finiteelement.py
        # It is u, v, w on the reference element
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
        # This matches the construction in https://github.com/firedrakeproject/ufl/blob/master/ufl/finiteelement/finiteelement.py
        # It is w, v, u on the reference element
        elif elem.family() == 'NCE':
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            variantlist.append([variant, variant, variant])
            contlist.append(['H1', 'H1', 'L2'])
            contlist.append(['H1', 'L2', 'H1'])
            contlist.append(['L2', 'H1', 'H1'])
            degreelist.append([degree, degree, degree-1])
            degreelist.append([degree, degree-1, degree])
            degreelist.append([degree-1, degree, degree])
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
    def __init__(self, elem):

        if isinstance(elem, TensorElement):
            self.ndofs = elem.value_size()  # do this BEFORE extracting baseelem
            elem = elem.sub_elements()[0]
        elif isinstance(elem, VectorElement):
            self.ndofs = elem.value_size()  # do this BEFORE extracting baseelem
            elem = elem.sub_elements()[0]
        else:
            self.ndofs = 1

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
                raise ValueError('Themis only supports TensorProductElements with the second element on an interval')
            if not (isinstance(elem1, FiniteElement) and isinstance(elem2, FiniteElement)):
                raise ValueError('Themis supports only TensorProductElements made of 2 FiniteElement')
            ncomp, variantlist, degreelist, contlist = flatten_tp_element(elem1, elem2)
        elif isinstance(elem, FiniteElement):
            ncomp, variantlist, degreelist, contlist = extract_element_info(elem)
        else:
            raise ValueError('Themis supports only FiniteElemet, EnrichedElement and TensorProductElement, along with Vector/Tensor Elements made from these')

        self.ncomp = ncomp
        self.interpolatory = True
        if self.ncomp > 1:
            self.interpolatory = False

        self.sub_elements = []

        self._maxdegree = 0

        self.nbasis = []
        self.nentries = []

        for ci in range(self.ncomp):
            self.sub_elements.append([])

            for degree, variant, cont in zip(degreelist[ci], variantlist[ci], contlist[ci]):

                self._maxdegree = max(degree, self._maxdegree)

                if not ((variant in ['mse', 'feec', 'mgd'] and cont == 'H1') or (variant == 'feec' and cont == 'L2')):
                    self.interpolatory = False

                self.sub_elements[ci].append(IntervalElement(variant, cont, degree, basis=False))

            # add "extra" elements in - those allow code generation to avoid dimension dependent logic
            while len(self.sub_elements[ci]) < 3:
                self.sub_elements[ci].append(IntervalElement('feec', 'L2', 0, basis=False))

            self.nbasis.append(self.sub_elements[ci][0].nbasis * self.sub_elements[ci][1].nbasis * self.sub_elements[ci][2].nbasis)
            self.nentries.append(self.sub_elements[ci][0].nentries * self.sub_elements[ci][1].nentries * self.sub_elements[ci][2].nentries)

        self.nentries_total = np.sum(np.array(self.nentries[ci]), dtype=np.int32)
        self.nbasis_total = np.sum(np.array(self.nbasis), dtype=np.int32)

    def dalist(self, name):
        dalist = ''
        for ci in range(self.ncomp):
            dalist = dalist + ',' + 'DM da_' + name + '_' + str(ci)
        return dalist

    def fieldargslist(self, name):
        fieldargs = ''
        for ci in range(self.ncomp):
            fieldargs = fieldargs + ',' + 'DM da_' + name + '_' + str(ci) + ',' + 'Vec ' + name + '_' + str(ci)
        return fieldargs

    def swidth(self):
        return self._maxdegree + 1
        # maxnbasis = max(max(self._nbasis))
        # return maxnbasis


class IntervalElement():

    def __init__(self, variant, cont, degree, symb=None, spts=None, basis=True):

        assert variant in ['feec', 'mgd', 'mse', 'qb']
        assert cont in ['H1', 'L2']

        self.degree = degree
        self.variant = variant
        self.cont = cont

        if spts is None:
            if variant == 'mse' and cont == 'L2':  # this accounts for the fact that DMSE is built using CMSE basis from complex, which counts from 1!
                spts = QuadratureNumerical('gll', degree+2).pts
            else:
                spts = QuadratureNumerical('gll', degree+1).pts  # count starts at 0
        self.spts = spts

        if basis:
            if symb is None:
                symb = sympy.var('x')
            self.symb = symb

        # compute offsets and offset mults
        if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
            of, om = _CG_offset_info(degree)
        if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
            of, om = _DG_offset_info(degree)
        if variant == 'mgd' and cont == 'H1':
            of, om = _GD_offset_info(degree)
        if variant == 'mgd' and cont == 'L2':
            of, om = _DGD_offset_info(degree)

        # compute entries
        if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
            ofe, ome = _CG_entries_info(degree)
        if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
            ofe, ome = _DG_entries_info(degree)
        if variant == 'mgd' and cont == 'H1':
            ofe, ome = _GD_entries_info(degree)
        if variant == 'mgd' and cont == 'L2':
            ofe, ome = _DGD_entries_info(degree)

        # compute basis and deriv functions
        if basis:
            if variant == 'feec' and cont == 'H1':
                b, d, d2 = _CG_basis(degree, symb, spts)
            if variant == 'feec' and cont == 'L2':
                b, d, d2 = _DG_basis(degree, symb, spts)
            if variant == 'mgd' and cont == 'H1':
                b, d, d2 = _GD_basis(degree, symb)
            if variant == 'mgd' and cont == 'L2':
                b, d, d2 = _DGD_basis(degree, symb)
            if variant == 'qb' and cont == 'H1':
                b, d, d2 = _CQB_basis(degree, symb)
            if variant == 'qb' and cont == 'L2':
                b, d, d2 = _DQB_basis(degree, symb)
            if variant == 'mse' and cont == 'H1':
                b, d, d2 = _CMSE_basis(degree, symb, spts)
            if variant == 'mse' and cont == 'L2':
                b, d, d2 = _DMSE_basis(degree, symb, spts)  # Note that spts here is actually the H1 spts from the derham complex partner

        # compute ndofs and nblocks
        if variant in ['feec', 'mse', 'qb'] and cont == 'H1':
            ndofs = degree
            nblocks = 1
        if variant in ['feec', 'mse', 'qb'] and cont == 'L2':
            ndofs = degree+1
            nblocks = 1
        if variant == 'mgd' and cont == 'H1':
            ndofs = 1
            nblocks = degree
        if variant == 'mgd' and cont == 'L2':
            ndofs = 1
            nblocks = degree+1

        self.offsets = of
        self.offset_mult = om
        self.entries_offsets = ofe
        self.entries_offset_mult = ome

        if basis:
            self.basis = b
            self.derivs = d
            self.derivs2 = d2

        self.nentries = ofe.shape[0]
        self.nbasis = degree+1
        self.nblocks = nblocks
        self.ndofs_per_element = ndofs

        self.of = a_to_cinit_string(of)
        self.om = a_to_cinit_string(om)
        self.ofe = a_to_cinit_string(ofe)
        self.ome = a_to_cinit_string(ome)

        if basis:
            self.lambdified_basis = []
            self.lambdified_derivs = []
            self.lambdified_derivs2 = []
            for bi in range(self.nblocks):
                self.lambdified_basis.append([])
                self.lambdified_derivs.append([])
                self.lambdified_derivs2.append([])
                for i in range(self.nbasis):
                    self.lambdified_basis[bi].append(sympy.lambdify(self.symb, self.basis[bi][i], "numpy"))
                    self.lambdified_derivs[bi].append(sympy.lambdify(self.symb, self.derivs[bi][i], "numpy"))
                    self.lambdified_derivs2[bi].append(sympy.lambdify(self.symb, self.derivs2[bi][i], "numpy"))

    def tabulate_exact(self, x, derivorder):
        npts = len(x)
        tabulation = []
        for bi in range(self.nblocks):
            tab = sympy.zeros(npts, self.nbasis)
            if derivorder == 0:
                symfuncs = self.basis[bi]
            if derivorder == 1:
                symfuncs = self.derivs[bi]
            if derivorder == 2:
                symfuncs = self.derivs2[bi]
            for i in range(self.nbasis):
                for j in range(npts):
                    if (symfuncs[i] == 1.0 and derivorder == 0):  # check for the constant basis function
                        tab[j, i] = sympy.Rational(1, 1)
                    elif (symfuncs[i] == 0.0 and derivorder > 0):
                        tab[j, i] = sympy.Rational(1, 1) * 0
                    else:
                        tab[j, i] = symfuncs[i].subs(self.symb, x[j])
            tabulation.append(tab)
        return tabulation

    def tabulate_numerical(self, x, derivorder):
        npts = x.shape[0]
        tabulation = np.zeros((self.nblocks, npts, self.nbasis))
        if derivorder == 0:
            symfuncs = self.lambdified_basis
        if derivorder == 1:
            symfuncs = self.lambdified_derivs
        if derivorder == 2:
            symfuncs = self.lambdified_derivs2
        for bi in range(self.nblocks):
            for i in range(self.nbasis):
                if (symfuncs[bi][i] == 1.0 and derivorder == 0):  # check for the constant basis function
                    tabulation[bi, :, i] = 1.0
                elif (symfuncs[bi][i] == 0.0 and derivorder > 0):
                    tabulation[bi, :, i] = 0.0
                else:
                    tabulation[bi, :, i] = np.squeeze(symfuncs[bi][i](x))
        return tabulation

# Lagrange Elements


def _CG_basis(order, symb, spts):
    symbas = []
    derivs = []
    derivs2 = []
    for i in range(order+1):
        symbas.append(lagrange_poly_support(i, spts, symb))
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    return [symbas, ], [derivs, ], [derivs2, ]


def _DG_basis(order, symb, spts):
    symbas = []
    derivs = []
    derivs2 = []
    if order >= 1:
        for i in range(order+1):
            symbas.append(lagrange_poly_support(i, spts, symb))
            derivs.append(sympy.diff(symbas[i]))
            derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    else:
        spts = [sympy.Rational(1, 2), ]
        symbas.append(sympy.Rational(1, 1))
        derivs.append(sympy.Rational(1, 1) * 0)
        derivs2.append(sympy.Rational(1, 1) * 0)
    return [symbas, ], [derivs, ], [derivs2, ]


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


def _GD_basis(order, symb):
    symbas = []
    derivs = []
    derivs2 = []
    a = 2
    b = -order
    p = a * np.arange(order+1) + b
    p = sympy.Rational(1, 2) * p + sympy.Rational(1, 2)
    for i in range(0, order+1):
        basis = lagrange_poly_support(i, p, symb)
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
        return [symbasleft, symbas, symbasright], [derivsleft, derivs, derivsright], [derivs2left, derivs2, derivs2right]
# ADD SUPPORT HERE FOR MORE GENERAL FORMULAS...
    else:
        return [symbas]*order, [derivs]*order, [derivs2]*order

# uses the formulas from Hiemstra et. al 2014 to create a compatible DG basis from a given CG basis


def create_compatible_l2_basis(cg_symbas):
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


def _DGD_basis(order, symb):
    if order >= 1:
        cg_symbas, _, _ = _GD_basis(order+1, symb)
        symbas, derivs, derivs2 = create_compatible_l2_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2


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
    return expanded_offsets, expanded_offset_multiplier


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


def _CQB_basis(order, symb):
    symbas = bernstein_polys(order, symb)
    derivs = []
    derivs2 = []
    for i in range(len(symbas)):
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    return [symbas, ], [derivs, ], [derivs2, ]


def _DQB_basis(order, symb):
    if order >= 1:
        cg_symbas, _, _ = _CQB_basis(order+1, symb)
        symbas, derivs, derivs2 = create_compatible_l2_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2


# MSE ELEMENTS
_CMSE_basis = _CG_basis

# Here, spts argument is actually those used for the corresponding CG space!


def _DMSE_basis(order, symb, spts):
    if order >= 1:
        cg_symbas, _, _ = _CMSE_basis(order+1, symb, spts)
        symbas, derivs, derivs2 = create_compatible_l2_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2
