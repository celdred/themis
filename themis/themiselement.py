
def extract_element_info(elem):
    degree = elem.degree()
    variant = elem.variant()
    if variant is None:
        variant = 'feec'

    degreelist = []
    contlist = []
    
    if elem.cell().cellname() == 'interval':
        ncomp = 1
        degreelist.append([degree,])
        if elem.family() == 'Discontinuous Lagrange':
            elemname = 'DQ'
            dofmap = np.array((0, degree+1, 0, 0),dtype=np.int32)
            contlist.append(['L2',])
        elif elem.family() == 'Lagrange':
            elemname = 'Q'                
            dofmap = np.array((1, degree-1, 0, 0),dtype=np.int32)
            contlist.append(['H1',])
        else:
            raise ValueError('themis supports only CG/DG on intervals')
        
    elif elem.cell().cellname() == 'quadrilateral':
        elemname = elem.family()                
        if elem.family() in ['DQ', 'Q']:
            degreelist.append([degree, degree])
            if elem.family() == 'DQ':
                contlist.append(['L2', 'L2'])
                dofmap = np.array((0, 0, (degree+1)*(degree+1), 0),dtype=np.int32)
            if elem.family() == 'Q':
                contlist.append(['H1', 'H1'])
                dofmap = np.array((1, degree-1, (degree-1)*(degree-1), 0),dtype=np.int32)
            ncomp = 1
        elif elem.family() == 'RTCF':
            dofmap = np.array((0, degree, 2*degree*(degree-1), 0),dtype=np.int32)
            ncomp = 2
            degreelist.append([degree, degree-1])
            degreelist.append([degree-1, degree])
            contlist.append(['H1', 'L2'])
            contlist.append(['L2', 'H1'])
        elif elem.family() == 'RTCE':
            dofmap = np.array((0, degree, 2*degree*(degree-1), 0),dtype=np.int32)
            ncomp = 2
            degreelist.append([degree-1, degree])
            degreelist.append([degree, degree-1])
            contlist.append(['L2', 'H1'])
            contlist.append(['H1', 'L2'])
        else:
            raise ValueError('themis supports only Q/DQ/RTCF/RTCE on quads')


    elif elem.cell().cellname() == 'hexahedron':
        elemname = elem.family()
        if elem.family() in ['DQ', 'Q']:
            degreelist.append([degree, degree, degree])
            if elem.family() == 'DQ':
                dofmap = np.array((0,0,0,(degree+1)*(degree+1)*(degree+1)),dtype=np.int32)
                contlist.append(['L2', 'L2', 'L2'])
            if elem.family() == 'Q':
                dofmap = np.array((1, degree-1, (degree-1)*(degree-1), (degree-1)*(degree-1)*(degree-1)),dtype=np.int32)
                contlist.append(['H1', 'H1', 'H1'])
            ncomp = 1
        elif elem.family() == 'NCF':
            dofmap = np.array((0, 0, degree*degree, 3*degree*degree*(degree-1)),dtype=np.int32)
            ncomp = 3
            degreelist.append([degree, degree-1, degree-1])
            degreelist.append([degree-1, degree, degree-1])
            degreelist.append([degree-1, degree-1, degree])
            contlist.append(['H1', 'L2', 'L2'])
            contlist.append(['L2', 'H1', 'L2'])
            contlist.append(['L2', 'L2', 'H1'])
        elif elem.family() == 'NCE':
            dofmap = np.array((0, degree, 2*degree*(degree-1), 3*degree*(degree-1)),dtype=np.int32)
            ncomp = 3
            degreelist.append([degree-1, degree, degree])
            degreelist.append([degree, degree-1, degree])
            degreelist.append([degree, degree, degree-1])
            contlist.append(['L2', 'H1', 'H1'])
            contlist.append(['H1', 'L2', 'H1'])
            contlist.append(['H1', 'H1', 'L2'])
        else:
            raise ValueError('themis supports only Q/DQ/NCF/NCE on hexahedrons')
    else:
        raise ValueError('Themis does not support cell type %s',elem.cell())
        
    return elemname,elem.cell().cellname(),degree,variant,ncomp,dofmap,degreelist,contlist

# DOESN'T SUPPORT TP/ENRICHED YET...
class ThemisElement():
    def __init__(self, elem, sptsh1=None, sptsl2=None):
        
        # ADD SUPPORT FOR TENSOR ELEMENTS
        if isinstance(elem, VectorElement):
            self.ndofs = elem.value_size()
            elem = elem.sub_elements()[0]
        else:
            self.ndofs = 1
        self.elem = elem
        
        self.elemname, self.cellname, self.degree, self.variant, self.ncomp, self.dofmap, degreelist, contlist = extract_element_info(elem)
        
        if self.variant in ['feec','mse']:
            self.sptsH1 = sptsh1 or ThemisQuadratureNumerical('gll', [self.degree+1]).get_pts()[0]
            if self.elemname in ['DQ','Q']:
                self.sptsL2 = sptsl2 or ThemisQuadratureNumerical('gll', [self.degree+1]).get_pts()[0] # DG counts from 0
            if self.elemname in ['RTCE','RTCF','NCE','NCF']:
                self.sptsL2 = sptsl2 or ThemisQuadratureNumerical('gll', [self.degree]).get_pts()[0] # lowest order RT is order 1, which uses DG0!

        if self.variant in ['feec','mse','qb']:
            self.nblocks = 1
            self.swidth = 1
        if self.variant in ['mgd',]:
            if self.elemname in ['DQ',]: # DG counts from 0
                self.nblocks = self.degree + 1
                self.swidth = self.degree//2 + 1
            else:
                self.nblocks = self.degree
                self.swidth = (self.degree-1)//2 + 1

        if (self.elemname == 'Q' and variant in ['feec','mse','mgd']) or (self.elemname == 'DQ' and self.variant == 'feec'):
            self.interpolatory = True
        else:
            self.interpolatory = False

        # Auto-generate rest of element info from a list of degrees and continuity types
        self.nbasis = []
        self.basis = []
        self.derivs = []
        self.derivs2 = []
        offsets = []
        
        self.location = []
        self.dofnum = []
        self.cellx = []
        self.celly = []
        self.cellz = []
        
        for ci in range(self.ncomp):

            self.nbasis.append([])
            self.basis.append([])
            self.derivs.append([])
            self.derivs2.append([])
            offsets.append([])
            for degree, cont in zip(degreelist[ci],contlist[ci]):
                
                # compute number of shape functions in each direction (=nbasis)
                self.nbasis[ci].append(degree + 1)

                # compute basis and deriv functions
                if cont == 'H1' and variant == 'feec': b, d, d2 = _CG_basis(degree, sptsH1)
                if cont == 'L2' and variant == 'feec': b, d, d2 = _DG_basis(degree, sptsL2)
                if cont == 'H1' and variant == 'mgd': b, d, d2 = _GD_basis(degree)
                if cont == 'L2' and variant == 'mgd': b, d, d2 = _DGD_basis(degree)
                if cont == 'H1' and variant == 'qb': b, d, d2 = _CQB_basis(degree)
                if cont == 'L2' and variant == 'qb': b, d, d2 = _DQB_basis(degree)
                if cont == 'H1' and variant == 'mse': b, d, d2 = _CMSE_basis(degree, sptsH1)
                if cont == 'L2' and variant == 'mse': b, d, d2 = _DMSE_basis(degree, sptsH1)  # Note use of sptsH1 here, since DMSE basis is built from CMSE basis!
   
                self.basis[ci].append(b)
                self.derivs[ci].append(d)
                self.derivs2[ci].append(d2)
                
                # compute offsets
                if cont == 'H1' and variant in ['feec','mse','qb']: o = _CG_offsets(degree)
                if cont == 'L2' and variant in ['feec','mse','qb']: o = _DG_offsets(degree)
                if cont == 'H1' and variant in ['mgd']: o = _CGD_offsets(degree)
                if cont == 'L2' and variant in ['mgd']: o = _DGD_offsets(degree)
                
                offsets[ci].append(o)
                
# ADD EMPTY ELEMENT BASIS/DERIVS? ALTHOUGH THIS IS ACTUALLY ONLY REALLY USED FOR EVAL, WHICH IS GOING AWAY...
        # Add "empty element" nbasis and offsets stuff- useful to let assembly template treat everything as 3D...

            while len(self.nbasis[ci]) < 3:
                self.nbasis[ci].append(1)
# THIS MIGHT BREAK WITH NBLOCKS STUFF?
                self.offsets[ci].append([0,],[PETSc.DMStag.StencilLocation.ELEM,])
            
            # compute location, dofnum and cell
            location = np.zeros((self.nbasis[ci][0],self.nbasis[ci][1],self.nbasis[ci][2]),dtype=np.int32)
            dofnum = np.zeros((self.nbasis[ci][0],self.nbasis[ci][1],self.nbasis[ci][2]),dtype=np.int32)

# THIS MIGHT BREAK WITH NBLOCKS STUFF?
            cellx, locx = offsets[ci][0]
            celly, locy = offsets[ci][1]
            cellz, locz = offsets[ci][2]
            self.cellx.append(cellx)
            self.celly.append(celly)
            self.cellz.append(cellz)
            
            dofdict = {}
            locdict = {}
            if self.cellname == 'interval':
                locdict[(PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.LEFT
                locdict[(PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.ELEM
                locdict[(PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.RIGHT
                dofdict[PETSc.DMStag.StencilLocation.LEFT] = 0
                dofdict[PETSc.DMStag.StencilLocation.ELEM] = 0
                dofdict[PETSc.DMStag.StencilLocation.RIGHT] = 0
            if self.cellname == 'quadrilateral':
                locdict[(PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.DOWN_LEFT
                locdict[(PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.DOWN
                locdict[(PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.DOWN_RIGHT
                locdict[(PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.LEFT
                locdict[(PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.ELEM
                locdict[(PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.RIGHT
                locdict[(PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.LEFT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.UP_LEFT
                locdict[(PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.ELEM,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.UP
                locdict[(PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.RIGHT,PETSc.DMStag.StencilLocation.ELEM)] = PETSc.DMStag.StencilLocation.UP_RIGHT
                dofdict[PETSc.DMStag.StencilLocation.DOWN_LEFT] = 0
                dofdict[PETSc.DMStag.StencilLocation.DOWN] = 0
                dofdict[PETSc.DMStag.StencilLocation.DOWN_RIGHT] = 0
                dofdict[PETSc.DMStag.StencilLocation.LEFT] = 0
                dofdict[PETSc.DMStag.StencilLocation.ELEM] = 0
                dofdict[PETSc.DMStag.StencilLocation.RIGHT] = 0
                dofdict[PETSc.DMStag.StencilLocation.UP_LEFT] = 0
                dofdict[PETSc.DMStag.StencilLocation.UP] = 0
                dofdict[PETSc.DMStag.StencilLocation.UP_RIGHT] = 0
 # IMPLEMENT THIS!
            if self.cellname == 'hexahedral':
                raise NotImplementedError('hexahedral elements not quite done yet...')            
            
# THIS WILL BREAK WITH NBLOCKS STUFF
# UGGGH REALLY NEED AN OUTER NBLOCKS LOOP?
            for lx in range(self.nbasis[ci][0]):
                for ly in range(self.nbasis[ci][1]):
                    for lz in range(self.nbasis[ci][2]):
                        loc = locdict[(locx[lx],locy[ly],locz[lz])]
                        location[lx][ly][lz] = loc
                        dofnum[[lx][ly][lz] = dofdict[loc]
                        dofdict[loc] = dofdict[loc] + 1
# THIS IS STILL BROKEN FOR NDOFS > 1
# SHOULD INTERFACE WITH HOW ASSEMBLE.TEMPLATE DOES LOOPS WITH D...
# BASICALLY HOW ARE VALS LAID OUT IN KERNELS (AND HOW ARE FIELDS FED IN)?
            
            self.dofnum.append(dofnum)
            self.location.append(location)
            

        self.nbasis = np.array(self.nbasis,dtype=np.int32)
        


    def get_sym_basis(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        blist = []
        for bi in range(self.nblocks):
            blist.append([])
            for basis in self.basis[ci][direc][bi]:
                adjbasis = basis.subs(xsymb, newsymb)
                blist[bi].append(adjbasis)
        return blist

    def get_sym_derivs(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for bi in range(self.nblocks):
            dlist.append([])
            for deriv in self.derivs[ci][direc][bi]:
                adjderiv = deriv.subs(xsymb, newsymb)
                dlist[bi].append(adjderiv)
        return dlist

    def get_sym_derivs2(self, ci, direc, newsymb):
        xsymb = sympy.var('x')
        dlist = []
        for bi in range(self.nblocks):
            dlist.append([])
            for deriv2 in self.derivs2[ci][direc][bi]:
                adjderiv2 = deriv2.subs(xsymb, newsymb)
                dlist[bi].append(adjderiv2)
        return dlist

    def get_basis_exact(self, ci, direc, x):
        xsymb = sympy.var('x')
        npts = len(x)
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_funcs = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self.basis[ci][direc][bi]
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
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_derivs = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self.derivs[ci][direc][bi]
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
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_derivs2 = sympy.zeros(nblocks, npts, nbasis)
        for bi in range(nblocks):
            symbas = self.derivs2[ci][direc][bi]
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
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_funcs = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self.basis[ci][direc][bi]
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
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_derivs = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self.derivs[ci][direc][bi]
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
        nbasis = len(self.basis[ci][direc][0])
        nblocks = self.nblocks
        basis_derivs2 = np.zeros((nblocks, npts, nbasis))
        for bi in range(nblocks):
            symbas = self.derivs2[ci][direc][bi]
            for i in range(len(symbas)):
                if symbas[i] == 0.0:  # check for the constant basis function
                    basis_derivs2[bi, :, i] = 0.0
                else:
                    deriv2 = sympy.lambdify(xsymb, symbas[i], "numpy")
                    basis_derivs2[bi, :, i] = np.squeeze(deriv2(x))
        return basis_derivs2

    def get_nbasis_info(self):
        return self.nbasis[:,0], self.nbasis[:,1], self.nbasis[:,2], np.prod(self.nbasis,axis=1), np.prod(self.nbasis,axis=None)

    def get_offset_info(self):
        return self.cellx, self.celly, self.cellz, self.location, self.dofnum

# SHOULD THIS EXIST?
# YES I THINK SO...
# ALSO WHAT ABOUT ENRICHED?

# Useful TP are Q/DQ/RTE/RTC analogues
# Plus scalar combinations that correspond to vertical or horizontal components of RTE and RTC
class ThemisElementTP(ThemisElement):
    def __init__(self,elem,hsptsh1=None, hsptsl2=None,vsptsh1=None,vsptsl2=None):
        
        pass
        
        # self.ncomp = 
        # self.helemname = 
        # self.velemname = 
        # self.hvariant = 
        # self.hdegree =
        # self.vvariant = 
        # self.vdegree =




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
    return [symbas, ], [derivs, ], [derivs2, ]


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
    return [symbas, ], [derivs, ], [derivs2, ]

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
        return [symbas]*order, [derivs]*order, [derivs2]*order

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
    if order >= 1:
        cg_symbas, _, _, _ = _GD_basis(order+1, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2



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

def _CQB_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    symbas = bernstein_polys(order, xsymb)
    derivs = []
    derivs2 = []
    for i in range(len(symbas)):
        derivs.append(sympy.diff(symbas[i]))
        derivs2.append(sympy.diff(sympy.diff(symbas[i])))
    return [symbas, ], [derivs, ], [derivs2, ]


def _DQB_basis(order, symb=None):
    xsymb = symb or sympy.var('x')
    if order >= 1:
        cg_symbas, _, _, _ = _CQB_basis(order+1, symb=xsymb)
        symbas, derivs, derivs2 = create_compatible_basis(cg_symbas)
    else:
        symbas = [[sympy.Rational(1, 1), ], ]
        derivs = [[sympy.Rational(1, 1) * 0, ], ]
        derivs2 = [[sympy.Rational(1, 1) * 0, ], ]
    return symbas, derivs, derivs2


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
    return symbas, derivs, derivs2


# FIX THESE
# TAKE INTO ACCOUNT BLOCKS!

def _CG_offsets(degree):
    pass
    
def _DG_offsets(degree):
    pass
    
def _CGD_offsets(degree):
    pass
    
def _DGD_offsets(degree):
    pass
    
