import numpy as np
# from ufl import VectorElement, TensorProductElement, EnrichedElement, HDivElement, HCurlElement, interval, quadrilateral
from ufl import VectorElement
import sympy
from lagrange import lagrange_poly_support  # gauss_lobatto,
from quadrature import ThemisQuadratureNumerical

# def check_continuous(family):
# return family == 'CG' or family == 'CGD' or family == 'CQB' or family == 'CMSE'

# def check_discontinuous(family):
# return family == 'DG' or family == 'DGD' or family == 'DQB' or family == 'DMSE'


class ThemisElement():
    def __init__(self, elem):

        # THESE SPTS SHOULD REALLY BE DIFFERENT VARIANTS I THINK...
        # THIS IS USEFUL MOSTLY FOR DISPERSION STUFF
        # BUT I THINK REALLY IMPLEMENTING A BUNCH OF VARIANTS IS THE BEST APPROACH

        # extract the "base" element for VectorElements
        # also get ndofs

        if isinstance(elem, VectorElement):
            self._ndofs = elem.value_size()  # do this BEFORE extracting baseelem
            elem = elem.sub_elements()[0]
        else:
            self._ndofs = 1

        # check that we are either a 1D "CG"/"DG" element, OR a Tensor Product of these elements, OR an enriched element made up of Hdiv/Hcurl that are made of TP of 1D elements

        # oneD = False
        # if elem.cell().num_vertices() == 2: #we are on an interval
        # oneD = True

# REVISE THIS
# NEED TO CHECK THAT WE ARE EITHER
# 1) AN ACCEPTABLE BASE ELEMENT: Q/DQ in 1D, or Q/DQ/RTCF/RTCE in 2D
# 2) OR A TENSOR PRODUCT OF A BASE ELEMENT WITH Q/DQ (POSSIBLY WITH HDIV/HCURL WRAPPERS)
# 3) OR AN ENRICHED ELEMENT MADE UP OF SUMS OF THESE

        # if (elem.cell().num_vertices() == 2) and (check_continuous(elem.family()) or check_discontinuous(elem.family())): #1D
        # oneD = True
        # elif isinstance(elem,TensorProductElement):
            # for subelem in elem.sub_elements():
            # if (subelem.cell().num_vertices() == 2) and (check_continuous(subelem.family()) or check_discontinuous(subelem.family())):
            # pass
            # else:
            # raise TypeError("Themis does not support Tensor Product Elements made with anything other than 1D CG/DG")
        # elif isinstance(elem,EnrichedElement):
            # for subelem in elem._elements:
            # if isinstance(subelem,HDivElement) or isinstance(subelem,HCurlElement):
            # subelem = subelem._element
            # if not isinstance(subelem,TensorProductElement):
            # raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements)  or TensorProductElements made of 1D CG/DG elements")
            # for subsubelem in subelem.sub_elements():
            # if (subsubelem.cell().num_vertices() == 2) and (check_continuous(subsubelem.family()) or check_discontinuous(subsubelem.family())):
            # pass
            # else:
            # raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements) or TensorProductElements made of 1D CG/DG elements")
            # if isinstance(subelem,TensorProductElement):
            # for subsubelem in subelem.sub_elements():
            # if (subsubelem.cell().num_vertices() == 2) and (check_continuous(subsubelem.family()) or check_discontinuous(subsubelem.family())):
            # pass
            # else:
            # raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements) or TensorProductElements made of 1D CG/DG elements")
            # else:
            # raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements)")
        # else:
            # raise TypeError("Themis supports only Tensor Product elements made with 1D CG/DG, or Enriched (possibly HDiv/HCurl) versions of these (except for 1D CG/DG)")

        # create list of subelements

        # THIS IS NOW BROKEN
        # self._subelemlist = []
        # if oneD:
            # self._subelemlist.append([elem,])
            # self._ncomp = 1
        # elif isinstance(elem,EnrichedElement):
            # for ci in range(len(elem._elements)):
            # self._subelemlist.append([])
            # if isinstance(elem._elements[ci],HDivElement) or isinstance(elem._elements[ci],HCurlElement):
            # subelements = elem._elements[ci]._element.sub_elements()
            # if isinstance(elem._elements[ci],TensorProductElement):
            # subelements = elem._elements[ci].sub_elements()
            # for subelem in subelements:
            # self._subelemlist[ci].append(subelem)
            # self._ncomp = len(elem._elements)

        # elif isinstance(elem,TensorProductElement):
            # self._subelemlist.append([])
            # subelements = elem.sub_elements()
            # for subelem in subelements:
            # self._subelemlist[0].append(subelem)
            # self._ncomp = 1

        # print(elem)
        # print(elem.family())
        # print(elem.variant())
        # print(elem.degree())
        # print(self._ndofs)
        # print(type(elem.variant()))

        family_to_continuity = {}
        family_to_continuity['DQ'] = 'L2'
        family_to_continuity['Discontinuous Lagrange'] = 'L2'
        family_to_continuity['Q'] = 'H1'
        family_to_continuity['CG'] = 'H1'
        family_to_continuity['Lagrange'] = 'H1'
        variant_to_elemname = {}
        variant_to_elemname['feecH1'] = 'CG'
        variant_to_elemname['feecL2'] = 'DG'
        variant_to_elemname['mgdH1'] = 'CGD'
        variant_to_elemname['mgdL2'] = 'DGD'
        # variant_to_elemname['chrisH1'] = 'CG' #REMOVE
        # variant_to_elemname['chrisL2'] = 'DG' #REMOVE
        variant_to_elemname['H1'] = 'CG'
        variant_to_elemname['L2'] = 'DG'

        degree = elem.degree()
        variant = elem.variant()

        # print(variant,elem.family(),degree)

        self._cont = family_to_continuity[elem.family()]
        self._elemnamelist = []
        self._degreelist = []
        self._contlist = []

        # if elem.variant() is None:
        # variant = ''
        if elem.cell().cellname() == 'interval':
            self._ncomp = 1
            cont = family_to_continuity[elem.family()]
            self._degreelist.append([degree, ])
            self._contlist.append([cont, ])
            self._elemnamelist.append([variant_to_elemname[variant + cont], ])

        if elem.cell().cellname() == 'quadrilateral':
            if elem.family() == 'DQ' or elem.family() == 'Q':
                self._ncomp = 1
                cont = family_to_continuity[elem.family()]
                self._degreelist.append([degree, degree])
                self._contlist.append([cont, cont])
                self._elemnamelist.append([variant_to_elemname[variant + cont], variant_to_elemname[variant + cont]])
            if elem.family() == 'RTCF':
                self._ncomp = 2
                self._degreelist.append([degree, degree-1])
                self._degreelist.append([degree-1, degree])
                self._contlist.append(['H1', 'L2'])
                self._contlist.append(['L2', 'H1'])
                self._elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
                self._elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
            if elem.family() == 'RTCE':
                self._ncomp = 2
                self._degreelist.append([degree-1, degree])
                self._degreelist.append([degree, degree-1])
                self._contlist.append(['L2', 'H1'])
                self._contlist.append(['H1', 'L2'])
                self._elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
                self._elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])

        # FIX THIS
        if elem.cell().cellname() == 'tensorproduct':
            raise ValueError("tensor product cells not implemented yet!")

        self._nbasis = []
        self._ndofs_per_element = []
        self._offsets = []
        self._offset_mult = []
        self._basis = []
        self._derivs = []
        self._derivs2 = []
        self._spts = []
        for ci in range(self._ncomp):

            self._nbasis.append([])
            self._ndofs_per_element.append([])
            self._offsets.append([])
            self._offset_mult.append([])
            self._basis.append([])
            self._derivs.append([])
            self._derivs2.append([])
            self._spts.append([])
            for elemname, degree in zip(self._elemnamelist[ci], self._degreelist[ci]):

                # this works because degree has already been adjusted when setting up degreelist
                if variant == 'feec':
                    sptsL2 = ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]
                    sptsH1 = ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]

                # compute number of shape functions in each direction (=nbasis); and number of degrees of freedom per element
                self._nbasis[ci].append(degree + 1)
                if elemname == 'CG':
                    self._ndofs_per_element[ci].append(degree)
                if elemname == 'DG':
                    self._ndofs_per_element[ci].append(degree + 1)
                if elemname == 'CGD':
                    self._ndofs_per_element[ci].append(1)
                if elemname == 'DGD':
                    self._ndofs_per_element[ci].append(1)
                if elemname == 'CQB':
                    self._ndofs_per_element[ci].append(degree)
                if elemname == 'DQB':
                    self._ndofs_per_element[ci].append(degree + 1)
                if elemname == 'CMSE':
                    self._ndofs_per_element[ci].append(degree)
                if elemname == 'DMSE':
                    self._ndofs_per_element[ci].append(degree + 1)

                # compute offsets and offset mults
                if elemname == 'CG':
                    of, om = _CG_offset_info(degree)
                if elemname == 'DG':
                    of, om = _DG_offset_info(degree)
                if elemname == 'CGD':
                    of, om = _CGD_offset_info(degree)
                if elemname == 'DGD':
                    of, om = _DGD_offset_info(degree)
                if elemname == 'CQB':
                    of, om = _CG_offset_info(degree)  # MIGHT BE WRONG?
                if elemname == 'DQB':
                    of, om = _DG_offset_info(degree)  # MIGHT BE WRONG?
                if elemname == 'CMSE':
                    of, om = _CG_offset_info(degree)
                if elemname == 'DMSE':
                    of, om = _DG_offset_info(degree)

                self._offsets[ci].append(of)
                self._offset_mult[ci].append(om)

                # compute basis and deriv functions
                if elemname == 'CG':
                    b, d, d2, s = _CG_basis(degree, sptsH1)
                if elemname == 'DG':
                    b, d, d2, s = _DG_basis(degree, sptsL2)
                if elemname == 'CGD':
                    b, d, d2, s = _CGD_basis(degree)
                if elemname == 'DGD':
                    b, d, d2, s = _DGD_basis(degree)
                if elemname == 'CQB':
                    b, d, d2, s = _CQB_basis(degree)
                if elemname == 'DQB':
                    b, d, d2, s = _DQB_basis(degree)
                if elemname == 'CMSE':
                    b, d, d2, s = _CMSE_basis(degree, sptsH1)
                if elemname == 'DMSE':
                    b, d, d2, s = _DMSE_basis(degree, sptsL2)
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)

            # add EmptyElements in
            while len(self._basis[ci]) < 3:
                of, om = _DG_offset_info(0)
                b, d, d2, s = _DG_basis(0, [0.5, ])
                self._offsets[ci].append(of)
                self._offset_mult[ci].append(om)
                self._basis[ci].append(b)
                self._derivs[ci].append(d)
                self._derivs2[ci].append(d2)
                self._spts[ci].append(s)

    def get_continuity(self, ci, direc):
        return self._contlist[ci][direc]

    def get_nx(self, ci, direc, ncell, bc):
        fam = self._elemnamelist[ci][direc]
        if fam == 'CG' or fam == 'CQB' or fam == 'CMSE':
            nx = self._degreelist[ci][direc] * ncell
            if (not bc == 'periodic'):
                nx = nx + 1
        if fam == 'DG' or fam == 'DQB' or fam == 'DMSE':
            nx = (self._degreelist[ci][direc]+1) * ncell
        if fam == 'CGD':
            nx = ncell
            if (not bc == 'periodic'):
                nx = nx + 1
        if fam == 'DGD':
            nx = ncell
        return nx

    def ndofs(self):
        return self._ndofs

    def get_nbasis(self, ci, direc):
        return self._nbasis[ci][direc]

    def get_ndofs_per_element(self, ci, direc):
        return self._ndofs_per_element[ci][direc]

    def get_local_size(self):
        bprod = np.prod(self._nbasis)
        return [bprod, self._ndofs]

    def get_offsets(self, ci, direc):
        return self._offsets[ci][direc], self._offset_mult[ci][direc]

    def get_sym_basis(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        blist = []
        for basis in self._basis[ci][direc]:
            adjbasis = basis.subs(xsymb, newsymb)
            blist.append(adjbasis)
        return blist

    def get_sym_derivs(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for deriv in self._derivs[ci][direc]:
            adjderiv = deriv.subs(xsymb, newsymb)
            dlist.append(adjderiv)
        return dlist

    def get_sym_derivs2(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for deriv2 in self._derivs2[ci][direc]:
            adjderiv2 = deriv2.subs(xsymb, newsymb)
            dlist.append(adjderiv2)
        return dlist

    # THESE ARE IN ORDER BASIS,QUAD
    def get_basis_exact(self, ci, direc, x):

        xsymb = sympy.var('x')
        symbas = self._basis[ci][direc]
        basis_funcs = sympy.zeros(len(symbas), len(x))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 0.5:  # check for the constant basis function
                    basis_funcs[i, j] = sympy.Rational(1, 2)
                else:
                    basis_funcs[i, j] = symbas[i].subs(xsymb, x[j])
        return basis_funcs

    # THESE ARE IN ORDER BASIS,QUAD
    def get_derivs_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs[ci][direc]
        basis_derivs = sympy.zeros(len(symbas), len(x))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 0:  # check for the constant deriv function
                    basis_derivs[i, j] = 0
                else:
                    basis_derivs[i, j] = symbas[i].subs(xsymb, x[j])
        return basis_derivs

    # THESE ARE IN ORDER BASIS,QUAD
    def get_derivs2_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs2[ci][direc]
        basis_derivs2 = sympy.zeros(len(symbas), len(x))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 0:  # check for the constant deriv2 function
                    basis_derivs2[i, j] = 0
                else:
                    basis_derivs2[i, j] = symbas[i].subs(xsymb, x[j])
        return basis_derivs2

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_basis(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._basis[ci][direc]
        basis_funcs = np.zeros((x.shape[0], len(symbas)))
        for i in range(len(symbas)):
            if symbas[i] == 0.5:  # check for the constant basis function
                basis_funcs[:, i] = 0.5
            else:
                basis = sympy.lambdify(xsymb, symbas[i], "numpy")
                basis_funcs[:, i] = np.squeeze(basis(x))
        return basis_funcs

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_derivs(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs[ci][direc]
        basis_derivs = np.zeros((x.shape[0], len(symbas)))
        for i in range(len(symbas)):
            if symbas[i] == 0.0:  # check for the constant deriv function
                basis_derivs[:, i] = 0.0
            else:
                deriv = sympy.lambdify(xsymb, symbas[i], "numpy")
                basis_derivs[:, i] = np.squeeze(deriv(x))
        return basis_derivs

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_derivs2(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs2[ci][direc]
        basis_derivs2 = np.zeros((x.shape[0], len(symbas)))
        for i in range(len(symbas)):
            if symbas[i] == 0.0:  # check for the constant deriv2 function
                basis_derivs2[:, i] = 0.0
            else:
                deriv2 = sympy.lambdify(xsymb, symbas[i], "numpy")
                basis_derivs2[:, i] = np.squeeze(deriv2(x))
        return basis_derivs2

    def get_icells(self, ci, direc, ncell, bc, interior_facet):
        elemname = self._elemnamelist[ci][direc]
        degree = self._degreelist[ci][direc]
        if elemname == 'CG':
            return _CG_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'DG':
            return _DG_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'CGD':
            return _CGD_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'DGD':
            return _DGD_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'CQB':
            return _CG_interaction_cells(ncell, bc, interior_facet, degree)  # MIGHT BE WRONG?
        if elemname == 'DQB':
            return _DG_interaction_cells(ncell, bc, interior_facet, degree)  # MIGHT BE WRONG?
        if elemname == 'CMSE':
            return _CG_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'DMSE':
            return _DG_interaction_cells(ncell, bc, interior_facet, degree)

    def maxdegree(self):
        maxdeg = max(max(self._degreelist))
        if self._cont == 'L2':
            maxdeg = maxdeg + 1  # this deals with fact that for pure L2 spaces we have deg +1 ndofs!
        return maxdeg

    def get_info(self, ci, direc, x):
        b = self.get_basis(ci, direc, x)
        d = self.get_derivs(ci, direc, x)
        return b, d, self._offsets[ci][direc], self._offset_mult[ci][direc]

    def get_ncomp(self):
        return self._ncomp

#   LAGRANGE ELEMENTS


def _CG_basis(order, spts):
    xsymb = sympy.var('x')
    # if spts == None:
    # spts = gauss_lobatto(order+1)
# SCALES SPTS TO [0,1]
    # spts = 0.5 * spts + 0.5

    symbas = []
    derivs = []
    derivs2 = []
    for i in range(order+1):
        symbas.append(lagrange_poly_support(i, spts, xsymb))
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))

    return symbas, derivs, derivs2, spts


def _DG_basis(order, spts):
    xsymb = sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    if order >= 1:
        # if spts == None:
        # spts = gauss_lobatto(order+1)
        # SCALES SPTS TO [0,1]
        # spts = 0.5 * spts + 0.5
        for i in range(order+1):
            symbas.append(lagrange_poly_support(i, spts, xsymb))
            derivs.append(sympy.diff(symbas[i]))
            derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    else:
        # spts = [0,]
        # SCALES SPTS TO [0,1]
        # spts = [0.5,]
        symbas.append(sympy.Rational(1, 2))
        derivs.append(sympy.Rational(1, 1) * 0)
        derivs2.append(sympy.Rational(1, 1) * 0)

    return symbas, derivs, derivs2, spts


def _CG_offset_info(order):
    offsets = np.arange(0, order+1, dtype=np.int32)
    offset_multiplier = order * np.ones(offsets.shape, dtype=np.int32)
    # 'p*i','p*i+1'...'p*i+p'
    return offsets, offset_multiplier


def _DG_offset_info(order):
    offsets = np.arange(0, order+1, dtype=np.int32)
    offset_multiplier = (order+1) * np.ones(offsets.shape, dtype=np.int32)
    # 'p*i','p*i+1'...'p*i+p'
    return offsets, offset_multiplier


def _CG_interaction_cells(ncell, bc, interior_facet, order):
    if bc == 'periodic':
        off = 0
    else:
        off = 1
    interaction_cells = np.zeros((ncell*order + off, 2), dtype=np.int32)
    ilist = np.arange(0, interaction_cells.shape[0])
    rightmost_bound = np.floor_divide(ilist, order)   # floor(n/p)
    leftmost_bound = np.floor_divide(ilist-1, order)  # floor((n-1)/p)
    # no circular wrapping since we need the actual value in order to get differences in pre-alloc correct ie 2 - (-1) = 3
    if not interior_facet:
        interaction_cells[:, 0] = leftmost_bound
        interaction_cells[:, 1] = rightmost_bound
        if bc == 'nonperiodic':
            interaction_cells[0, 0] = interaction_cells[0, 1]
            interaction_cells[-1, 1] = interaction_cells[-1, 0]
    if interior_facet:
        interaction_cells[:, 0] = leftmost_bound - 1
        interaction_cells[:, 1] = rightmost_bound + 1
    return interaction_cells


def _DG_interaction_cells(ncell, bc, interior_facet, order):
    interaction_cells = np.zeros((ncell*(order+1), 2), dtype=np.int32)
    ilist = np.arange(0, interaction_cells.shape[0])
    rightmost_bound = np.floor_divide(ilist, order+1)  # floor(n/p)
    if not interior_facet:
        interaction_cells[:, 0] = rightmost_bound
        interaction_cells[:, 1] = rightmost_bound
    if interior_facet:
        interaction_cells[:, 0] = rightmost_bound - 1
        interaction_cells[:, 1] = rightmost_bound + 1
    return interaction_cells


#  MGD ELEMENTS


def _CGD_basis(order):
    xsymb = sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    a = 2
    b = -order
    p = a * np.arange(order+1) + b

# SCALES "SPTS" TO [0,1]
    p = 0.5 * p + 0.5
    # spts = [-1,1]
    spts = [0, 1]
    for i in range(0, order+1):
        basis = lagrange_poly_support(i, p, xsymb)
        symbas.append(basis)
        derivs.append(sympy.diff(basis))
        derivs2.append(sympy.diff(sympy.diff(basis)))

    return symbas, derivs, derivs2, spts


def _DGD_basis(order):
    xsymb = sympy.var('x')
    symbas = []
    derivs = []
    derivs2 = []
    cg_symbas = []

    a = 2
    b = -(order+1)
    p = a * np.arange(order+2) + b
# SCALES "SPTS" TO [0,1]
    p = 0.5 * p + 0.5
    # spts = [0,]
    spts = [0.5, ]
    for i in range(0, order+2):
        cg_symbas.append(lagrange_poly_support(i, p, xsymb))

    symbas.append(sympy.diff(-cg_symbas[0]))
    derivs.append(sympy.diff(symbas[0]))
    derivs2.append(sympy.diff(sympy.diff(symbas[0])))
    for i in range(1, order+1):
        dN = sympy.diff(cg_symbas[i])
        basis = symbas[i-1] - dN
        symbas.append(basis)
        derivs.append(sympy.diff(basis))
        derivs2.append(sympy.diff(sympy.diff(basis)))

    return symbas, derivs, derivs2, spts


def _CGD_offset_info(order):
    offsets = np.arange(-(order-1)//2, (order-1)//2+2, 1, dtype=np.int32)
    offset_multiplier = np.ones(offsets.shape, dtype=np.int32)
    return offsets, offset_multiplier


def _DGD_offset_info(order):
    offsets = np.arange(-(order)//2, (order)/2+1, 1, dtype=np.int32)  # THIS MIGHT BE WRONG!
    offset_multiplier = np.ones(offsets.shape, dtype=np.int32)
    return offsets, offset_multiplier

# ADD FACET INTEGRAL STUFF


def _CGD_interaction_cells(ncell, bc, interior_facet, order):
    if bc == 'periodic':
        off = 0
    else:
        off = 1
    leftmost = -(order-1)//2-1
    rightmost = (order-1)//2
    if interior_facet:
        leftmost = leftmost - 1
        rightmost = rightmost + 1
    interaction_offsets = np.array([leftmost, rightmost], dtype=np.int32)
    interaction_cells = np.zeros((ncell+off, 2), dtype=np.int32)
    ilist = np.arange(0, interaction_cells.shape[0])
    interaction_cells[:, :] = np.expand_dims(ilist, axis=1) + np.expand_dims(interaction_offsets, axis=0)

    return interaction_cells

# ADD FACET INTEGRAL STUFF


def _DGD_interaction_cells(ncell, bc, interior_facet, order):

    leftmost = -(order-1)//2
    rightmost = (order-1)//2
    if interior_facet:
        leftmost = leftmost - 1
        rightmost = rightmost + 1
    interaction_offsets = np.array([leftmost, rightmost], dtype=np.int32)
    interaction_cells = np.zeros((ncell, 2), dtype=np.int32)
    ilist = np.arange(0, interaction_cells.shape[0])
    interaction_cells[:, :] = np.expand_dims(ilist, axis=1) + np.expand_dims(interaction_offsets, axis=0)
    return interaction_cells


# QB ELEMENTS


def _CQB_basis(order):
    pass


def _DQB_basis(order):
    pass

# I THINK THESE ARE THE SAME AS CG/DG...BUT MAYBE INTERACTION CELLS IS DIFFERENT?


def _CQB_offset_info(order):
    pass


def _DQB_offset_info(order):
    pass


def _CQB_interaction_cells(ncell, bc, interior_facet, order):
    pass


def _DQB_interaction_cells(ncell, bc, interior_facet, order):
    pass

# MSE ELEMENTS


_CMSE_basis = _CG_basis


def _DMSE_basis(order, spts=None):
    pass
