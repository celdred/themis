import numpy as np
from ufl import VectorElement,TensorProductElement,EnrichedElement,HDivElement, HCurlElement, FiniteElement
import sympy
from lagrange import lagrange_poly_support  # gauss_lobatto,
from quadrature import ThemisQuadratureNumerical

from petscshim import PETSc

variant_to_elemname = {}
variant_to_elemname['feecH1'] = 'CG'
variant_to_elemname['feecL2'] = 'DG'
variant_to_elemname['mseH1'] = 'CMSE'
variant_to_elemname['mseL2'] = 'DMSE'
variant_to_elemname['qbH1'] = 'CQB'
variant_to_elemname['qbL2'] = 'DQB'
variant_to_elemname['mgdH1'] = 'CGD'
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
        variantlist.append([variant,])
        if elem.family() == 'Discontinuous Lagrange': 
            elemnamelist.append([variant_to_elemname[variant + 'L2'], ])
            contlist.append(['L2',])
        elif elem.family() == 'Lagrange': 
            elemnamelist.append([variant_to_elemname[variant + 'H1'], ])
            contlist.append(['H1',])
        else:
            raise ValueError('themis supports only CG/DG on intervals')
        
    elif elem.cell().cellname() == 'quadrilateral':
        if elem.family() in ['DQ','Q']:
            variantlist.append([variant,variant])
            degreelist.append([degree, degree])
            if elem.family() == 'DQ': 
                elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2']])
                contlist.append(['L2','L2',])
            if elem.family() == 'Q': 
                elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1']])
                contlist.append(['H1','H1',])
            ncomp = 1
        elif elem.family() == 'RTCF':
            variantlist.append([variant,variant])
            variantlist.append([variant,variant])
            degreelist.append([degree, degree-1])
            degreelist.append([degree-1, degree])
            contlist.append(['H1', 'L2'])
            contlist.append(['L2', 'H1'])
            elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'L2']])
            elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'H1']])
            ncomp = 2
        elif elem.family() == 'RTCE':
            variantlist.append([variant,variant])
            variantlist.append([variant,variant])
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
        if elem.family() in ['DQ','Q']:
            variantlist.append([variant,variant,variant])
            degreelist.append([degree, degree, degree])
            if elem.family() == 'DQ': 
                elemnamelist.append([variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2'], variant_to_elemname[variant + 'L2']])
                contlist.append(['L2', 'L2', 'L2'])
            if elem.family() == 'Q': 
                elemnamelist.append([variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1'], variant_to_elemname[variant + 'H1']])
                contlist.append(['H1', 'H1', 'H1'])
            ncomp = 1
        elif elem.family() == 'NCF':
            variantlist.append([variant,variant,variant])
            variantlist.append([variant,variant,variant])
            variantlist.append([variant,variant,variant])
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
            variantlist.append([variant,variant,variant])
            variantlist.append([variant,variant,variant])
            variantlist.append([variant,variant,variant])
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
        raise ValueError('Themis does not support this cell type')

    return ncomp,variantlist,degreelist,elemnamelist,contlist

def flatten_tp_element(elem1,elem2):
    ncomp1,variantlist1,degreelist1,elemnamelist1,contlist1 = extract_element_info(elem1)
    ncomp2,variantlist2,degreelist2,elemnamelist2,contlist2 = extract_element_info(elem2)
    for c1 in range(ncomp1):
        variantlist1[c1].append(variantlist2[0][0])
        degreelist1[c1].append(degreelist2[0][0])
        elemnamelist1[c1].append(elemnamelist2[0][0])
        contlist1[c1].append(contlist2[0][0])
    return ncomp1,variantlist1,degreelist1,elemnamelist1,contlist1
    
def merge_enriched_element(elem1,elem2):
    if isinstance(elem1,FiniteElement):
        ncomp1,variantlist1,degreelist1,elemnamelist1,contlist1 = extract_element_info(elem1)
    elif isinstance(elem1,TensorProductElement):
        subelem1,subelem2 = elem1.sub_elements()
        ncomp1,variantlist1,degreelist1,elemnamelist1,contlist1 = flatten_tp_element(subelem1,subelem2)
        
    if isinstance(elem2,FiniteElement):
        ncomp2,variantlist2,degreelist2,elemnamelist2,contlist2 = extract_element_info(elem2)
    elif isinstance(elem2,TensorProductElement):
        subelem1,subelem2 = elem2.sub_elements()
        ncomp2,variantlist2,degreelist2,elemnamelist2,contlist2 = flatten_tp_element(subelem1,subelem2)
    
    ncomp = ncomp1 + ncomp2
    variantlist = variantlist1 + variantlist2
    degreelist = degreelist1 + degreelist2
    elemnamelist = elemnamelist1 + elemnamelist2
    contlist = contlist1 + contlist2
    return ncomp,variantlist,degreelist,elemnamelist,contlist
    
class ThemisElement():
        # THESE SPTS SHOULD REALLY BE DIFFERENT VARIANTS I THINK...
        # THIS IS USEFUL MOSTLY FOR DISPERSION STUFF
        # BUT I THINK REALLY IMPLEMENTING A BUNCH OF VARIANTS IS THE BEST APPROACH
    def __init__(self, elem,sptsH1=None,sptsL2=None):

        
# ADD SUPPORT FOR TENSOR ELEMENTS
        if isinstance(elem, VectorElement):
            self._ndofs = elem.value_size()  # do this BEFORE extracting baseelem
            elem = elem.sub_elements()[0]
        else:
            self._ndofs = 1
            


        if isinstance(elem, EnrichedElement):
            elem1,elem2 =  elem._elements
            if (isinstance(elem1,HDivElement) and isinstance(elem2,HDivElement)) or (isinstance(elem1,HCurlElement) and isinstance(elem2,HCurlElement)):
                elem1 = elem1._element
                elem2 = elem2._element
            else:
                raise ValueError('Themis supports only EnrichedElement made of 2 HDiv/HCurl elements')
            if not ((isinstance(elem1,FiniteElement) or isinstance(elem1,TensorProductElement)) and (isinstance(elem2,FiniteElement) or isinstance(elem2,TensorProductElement))):
                raise ValueError('Themis supports only EnrichedElement made of 2 HDiv/HCurl elements that are themselves FiniteElement or TensorProductElement')
            ncomp,variantlist,degreelist,elemnamelist,contlist = merge_enriched_element(elem1,elem2)
        elif isinstance(elem, TensorProductElement):
            elem1,elem2 =  elem.sub_elements()
            if not (elem2.cell().cellname() == 'interval'):
                raise ValueError('Themis only supports tensor product elements with the second element on an interval')
            if not (isinstance(elem1,FiniteElement) and isinstance(elem2,FiniteElement)):
                raise ValueError('Themis supports only tensor product elements of FiniteElement')
            ncomp,variantlist,degreelist,elemnamelist,contlist = flatten_tp_element(elem1,elem2)
        elif isinstance(elem, FiniteElement):
            ncomp,variantlist,degreelist,elemnamelist,contlist = extract_element_info(elem)
        else:
            raise ValueError('Themis supports only FiniteElemet, EnrichedElement and TensorProductElement')
        
        # PETSc.Sys.Print(elem,ncomp)
        # PETSc.Sys.Print(variantlist)
        # PETSc.Sys.Print(degreelist)
        # PETSc.Sys.Print(elemnamelist)
        # PETSc.Sys.Print(contlist)
        
        self._cont = elem.sobolev_space()
        
        self._ncomp = ncomp
        self._elemnamelist = elemnamelist
        self._degreelist = degreelist
        self._contlist = contlist
        self._variantlist = variantlist

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
            for elemname, degree, variant in zip(self._elemnamelist[ci], self._degreelist[ci],self._variantlist[ci]):

                # this works because degree has already been adjusted when setting up degreelist
                if variant == 'feec': # REALLY HERE VARIANT SHOULD BE DONE MORE CLEVERLY IE VARIOUS OPTIONS OF SPTS...
                    if elemname in ['DG','DMSE']:
                        if (sptsL2 is None): spts = ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]
                        else: spts = sptsL2
                    if elemname in ['CG','DMSE']:
                        if (sptsH1 is None): spts = ThemisQuadratureNumerical('gll', [degree+1]).get_pts()[0]
                        else: spts = sptsH1
                        
                #PETSc.Sys.Print(ci,elemname,variant,degree)
                #PETSc.Sys.Print(spts)
                
                # compute number of shape functions in each direction (=nbasis); and number of degrees of freedom per element
                self._nbasis[ci].append(degree + 1)
                if elemname in ['CG','CQB','CMSE']:
                    self._ndofs_per_element[ci].append(degree)
                if elemname in ['DG','DQB','DMSE']:
                    self._ndofs_per_element[ci].append(degree + 1)
                if elemname == 'CGD':
                    self._ndofs_per_element[ci].append(1)
                if elemname == 'DGD':
                    self._ndofs_per_element[ci].append(1)


                # compute offsets and offset mults
                if elemname in ['CG','CQB','CMSE']:  # MIGHT BE WRONG FOR CQB?
                    of, om = _CG_offset_info(degree)
                if elemname in ['DG','DQB','DMSE']: # MIGHT BE WRONG FOR DQB?
                    of, om = _DG_offset_info(degree)
                if elemname == 'CGD':
                    of, om = _CGD_offset_info(degree)
                if elemname == 'DGD':
                    of, om = _DGD_offset_info(degree)
    
                self._offsets[ci].append(of)
                self._offset_mult[ci].append(om)

                # compute basis and deriv functions
                if elemname == 'CG':
                    b, d, d2, s = _CG_basis(degree, spts)
                if elemname == 'DG':
                    b, d, d2, s = _DG_basis(degree, spts)
                if elemname == 'CGD':
                    b, d, d2, s = _CGD_basis(degree)
                if elemname == 'DGD':
                    b, d, d2, s = _DGD_basis(degree)
                if elemname == 'CQB':
                    b, d, d2, s = _CQB_basis(degree)
                if elemname == 'DQB':
                    b, d, d2, s = _DQB_basis(degree)
                if elemname == 'CMSE':
                    b, d, d2, s = _CMSE_basis(degree, spts)
                if elemname == 'DMSE':
                    b, d, d2, s = _DMSE_basis(degree, spts)
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
        if fam in ['CG','CQB','CMSE']:
            nx = self._degreelist[ci][direc] * ncell
            if (not bc == 'periodic'):
                nx = nx + 1
        if fam in ['DG','DQB','DMSE']:
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

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_basis_exact(self, ci, direc, x):

        xsymb = sympy.var('x')
        symbas = self._basis[ci][direc]
        basis_funcs = sympy.zeros(len(x),len(symbas))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 1.0:  # check for the constant basis function
                    basis_funcs[j, i] = sympy.Rational(1, 1)
                else:
                    basis_funcs[j, i] = symbas[i].subs(xsymb, x[j])
        return basis_funcs

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_derivs_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs[ci][direc]
        basis_derivs = sympy.zeros(len(x),len(symbas))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 0:  # check for the constant deriv function
                    basis_derivs[j, i] = 0
                else:
                    basis_derivs[j, i] = symbas[i].subs(xsymb, x[j])
        return basis_derivs

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_derivs2_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._derivs2[ci][direc]
        basis_derivs2 = sympy.zeros(len(x),len(symbas))
        for i in range(len(symbas)):
            for j in range(len(x)):
                if symbas[i] == 0:  # check for the constant deriv2 function
                    basis_derivs2[j, i] = 0
                else:
                    basis_derivs2[j, i] = symbas[i].subs(xsymb, x[j])
        return basis_derivs2

    # NOTE ORDER HERE IS QUADPT, BASIS
    def get_basis(self, ci, direc, x):
        xsymb = sympy.var('x')
        symbas = self._basis[ci][direc]
        basis_funcs = np.zeros((x.shape[0], len(symbas)))
        for i in range(len(symbas)):
            if symbas[i] == 1.0:  # check for the constant basis function
                basis_funcs[:, i] = 1.0
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
        if elemname in ['CG','CQB','CMSE']:  # MIGHT BE WRONG FOR CQB?
            return _CG_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname in ['DG','DQB','DMSE']:  # MIGHT BE WRONG FOR DQB?
            return _DG_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'CGD':
            return _CGD_interaction_cells(ncell, bc, interior_facet, degree)
        if elemname == 'DGD':
            return _DGD_interaction_cells(ncell, bc, interior_facet, degree)

    def swidth(self):
        #maxdeg = max(max(self._degreelist))
        maxnbasis = max(max(self._nbasis))
        #maxnodfs = max(max(self._ndofs_per_element))
        
        #if self._cont == 'L2':
        #    maxdeg = maxdeg + 1  # this deals with fact that for pure L2 spaces we have deg +1 ndofs!
        #PETSc.Sys.Print(self._cont,maxdeg,maxnodfs,maxnbasis)
        #PETSc.Sys.Print(self._cont,swidth)
        #PETSc.Sys.Print(self._degreelist)
        return maxnbasis

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
        for i in range(order+1):
            symbas.append(lagrange_poly_support(i, spts, xsymb))
            derivs.append(sympy.diff(symbas[i]))
            derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    else:
        spts = [sympy.Rational(1,2),]
        symbas.append(sympy.Rational(1, 1))
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
    #print('cg',interaction_cells)
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
    #print('dg',interaction_cells)
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
    p = sympy.Rational(1,2) * p + sympy.Rational(1,2)
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
    p = sympy.Rational(1,2) * p + sympy.Rational(1,2)
    spts = [sympy.Rational(1,2), ]
    
    if order >=1:
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

    else:
        symbas.append(sympy.Rational(1, 1))
        derivs.append(sympy.Rational(1, 1) * 0)
        derivs2.append(sympy.Rational(1, 1) * 0)
        
    return symbas, derivs, derivs2, spts


def _CGD_offset_info(order):
    offsets = np.arange(-(order-1)//2, (order-1)//2+2, 1, dtype=np.int32)
    offset_multiplier = np.ones(offsets.shape, dtype=np.int32)
    return offsets, offset_multiplier


def _DGD_offset_info(order):
    offsets = np.arange(-(order)//2, (order)/2+1, 1, dtype=np.int32)
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
    #print('cgd',interaction_cells)
    return interaction_cells

# ADD FACET INTEGRAL STUFF


def _DGD_interaction_cells(ncell, bc, interior_facet, order):

    leftmost = -(order-1)//2
    rightmost = (order-1)//2+1 #ADDED 1 HERE
    if interior_facet:
        leftmost = leftmost - 1
        rightmost = rightmost + 1
    interaction_offsets = np.array([leftmost, rightmost], dtype=np.int32)
    interaction_cells = np.zeros((ncell, 2), dtype=np.int32)
    ilist = np.arange(0, interaction_cells.shape[0])
    interaction_cells[:, :] = np.expand_dims(ilist, axis=1) + np.expand_dims(interaction_offsets, axis=0)
    #print('dgd',interaction_cells)
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
