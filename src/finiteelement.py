import numpy as np
from ufl import VectorElement,TensorProductElement,EnrichedElement,HDivElement,HCurlElement,interval,quadrilateral
import sympy
from lagrange import gauss_lobatto,lagrange_poly_support

def check_continuous(family):
	return family == 'CG' or family == 'CGD' or family == 'CQB' or family == 'CMSE'
	
def check_discontinuous(family):
	return family == 'DG' or family == 'DGD' or family == 'DQB' or family == 'DMSE'
	
class ThemisElement():
	def __init__(self,elem,sptsH1=None,sptsL2=None): #THESE SPTS SHOULD REALLY BE DIFFERENT VARIANTS I THINK...
		
		#extract the "base" element for VectorElements
		#also get ndofs

		if isinstance(elem,VectorElement):
			self._ndofs = elem.value_size() #do this BEFORE extracting baseelem
			elem = elem.sub_elements()[0]
		else:
			self._ndofs = 1
			
		#check that we are either a 1D "CG"/"DG" element, OR a Tensor Product of these elements, OR an enriched element made up of Hdiv/Hcurl that are made of TP of 1D elements

		#oneD = False
		#if elem.cell().num_vertices() == 2: #we are on an interval
		#	oneD = True
		
		#REVISE THIS
		#if (elem.cell().num_vertices() == 2) and (check_continuous(elem.family()) or check_discontinuous(elem.family())): #1D
		#	oneD = True
		#elif isinstance(elem,TensorProductElement):
			#for subelem in elem.sub_elements():
				#if (subelem.cell().num_vertices() == 2) and (check_continuous(subelem.family()) or check_discontinuous(subelem.family())):
					#pass
				#else:
					#raise TypeError("Themis does not support Tensor Product Elements made with anything other than 1D CG/DG")
		#elif isinstance(elem,EnrichedElement):
			#for subelem in elem._elements:
				#if isinstance(subelem,HDivElement) or isinstance(subelem,HCurlElement):
					#subelem = subelem._element
					#if not isinstance(subelem,TensorProductElement):
						#raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements)  or TensorProductElements made of 1D CG/DG elements")
					#for subsubelem in subelem.sub_elements():
						#if (subsubelem.cell().num_vertices() == 2) and (check_continuous(subsubelem.family()) or check_discontinuous(subsubelem.family())):
							#pass
						#else:
							#raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements) or TensorProductElements made of 1D CG/DG elements")
				#if isinstance(subelem,TensorProductElement):
					#for subsubelem in subelem.sub_elements():
						#if (subsubelem.cell().num_vertices() == 2) and (check_continuous(subsubelem.family()) or check_discontinuous(subsubelem.family())):
							#pass
						#else:
							#raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements) or TensorProductElements made of 1D CG/DG elements")					
				#else:
					#raise TypeError("Themis does not support Enriched elements made with anything other than HDiv/HCurl elements (which are themselves Tensor Products of 1D CG/DG elements)")
		#else:
			#raise TypeError("Themis supports only Tensor Product elements made with 1D CG/DG, or Enriched (possibly HDiv/HCurl) versions of these (except for 1D CG/DG)")

		#create list of subelements
		
		#THIS IS NOW BROKEN
		#self._subelemlist = []
		#if oneD:
			#self._subelemlist.append([elem,])
			#self._ncomp = 1
		#elif isinstance(elem,EnrichedElement):
			#for ci in range(len(elem._elements)):
				#self._subelemlist.append([])
				#if isinstance(elem._elements[ci],HDivElement) or isinstance(elem._elements[ci],HCurlElement):
					#subelements = elem._elements[ci]._element.sub_elements()
				#if isinstance(elem._elements[ci],TensorProductElement):
					#subelements = elem._elements[ci].sub_elements()
				#for subelem in subelements:
					#self._subelemlist[ci].append(subelem)
			#self._ncomp = len(elem._elements)
				
		#elif isinstance(elem,TensorProductElement):
			#self._subelemlist.append([])
			#subelements = elem.sub_elements()
			#for subelem in subelements:
				#self._subelemlist[0].append(subelem)
			#self._ncomp = 1	

		#print(elem)
		#print(elem.family())
		#print(elem.variant())
		#print(elem.degree())
		#print(self._ndofs)
		#print(type(elem.variant()))
		
		#FIX THIS STUFF UP FOR MORE GENERAL ELEMENTS
		family_to_continuity = {}
		family_to_continuity['DQ'] = 'L2'
		family_to_continuity['Q'] = 'H1'
		variant_to_elemname = {}
		variant_to_elemname['chrisH1'] = 'CG'
		variant_to_elemname['chrisL2'] = 'DG'
		variant_to_elemname['H1'] = 'CG'
		variant_to_elemname['L2'] = 'DG'
		#####################
		
		self._elemnamelist = []
		self._degreelist = []
		self._contlist = []
		
		degree = elem.degree()
		variant = elem.variant()
		if elem.variant() == None: variant = '' #EVENTUALLY REMOVE THIS
		if elem.cell().cellname() == 'interval':
			self._ncomp = 1
			cont = family_to_continuity[elem.family()]
			self._degreelist.append([degree,])
			self._contlist.append([cont,])
			self._elemnamelist.append([variant_to_elemname[variant + cont],])
			
		if elem.cell().cellname() == 'quadrilateral':
			if elem.family() == 'DQ' or elem.family() == 'Q':
				self._ncomp = 1
				cont = family_to_continuity[elem.family()]
				self._degreelist.append([degree,degree])
				self._contlist.append([cont,cont])
				self._elemnamelist.append([variant_to_elemname[variant + cont],variant_to_elemname[variant + cont]])
			if elem.family() == 'RTCF':
				self._ncomp = 2
				self._degreelist.append([degree,degree-1])
				self._degreelist.append([degree-1,degree])
				self._contlist.append(['H1','L2'])
				self._contlist.append(['L2','H1'])
				self._elemnamelist.append([variant_to_elemname[variant + 'H1'],variant_to_elemname[variant + 'L2']])
				self._elemnamelist.append([variant_to_elemname[variant + 'L2'],variant_to_elemname[variant + 'H1']])				
			if elem.family() == 'RTCE':
				self._ncomp = 2
				self._degreelist.append([degree-1,degree])
				self._degreelist.append([degree,degree-1])
				self._contlist.append(['L2','H1'])
				self._contlist.append(['H1','L2'])
				self._elemnamelist.append([variant_to_elemname[variant + 'L2'],variant_to_elemname[variant + 'H1']])				
				self._elemnamelist.append([variant_to_elemname[variant + 'H1'],variant_to_elemname[variant + 'L2']])

		#FIX THIS
		if elem.cell().cellname() == 'tensorproduct':
			raise ValueError("tensor product cells not implemented yet!")
		
		self._nbasis = []
		self._ndofs_per_element = []
		self._offsets = []
		self._offset_mult = []
		self._basis = []
		self._derivs = []
		self._spts = []
		for ci in range(self._ncomp):
			self._nbasis.append([])
			self._ndofs_per_element.append([])
			self._offsets.append([])
			self._offset_mult.append([])
			self._basis.append([])
			self._derivs.append([])
			self._spts.append([])
			for elemname,degree in zip(self._elemnamelist[ci],self._degreelist[ci]):
				
				#compute number of shape functions in each direction (=nbasis); and number of degrees of freedom per element
				self._nbasis[ci].append(degree + 1)
				if elemname == 'CG': self._ndofs_per_element[ci].append(degree)
				if elemname == 'DG': self._ndofs_per_element[ci].append(degree + 1)
				if elemname == 'CGD': self._ndofs_per_element[ci].append(1)
				if elemname == 'DGD': self._ndofs_per_element[ci].append(1)
				if elemname == 'CQB': self._ndofs_per_element[ci].append(degree)
				if elemname == 'DQB': self._ndofs_per_element[ci].append(degree + 1)
				if elemname == 'CMSE': self._ndofs_per_element[ci].append(degree)
				if elemname == 'DMSE': self._ndofs_per_element[ci].append(degree + 1)
				
				#compute offsets and offset mults
				if elemname == 'CG': of,om = _CG_offset_info(degree)
				if elemname == 'DG': of,om = _DG_offset_info(degree)
				if elemname == 'CGD': of,om = _CGD_offset_info(degree) 
				if elemname == 'DGD': of,om = _DGD_offset_info(degree) 
				if elemname == 'CQB': of,om = _CG_offset_info(degree) #MIGHT BE WRONG?
				if elemname == 'DQB': of,om = _DG_offset_info(degree) #MIGHT BE WRONG?
				if elemname == 'CMSE': of,om = _CG_offset_info(degree)
				if elemname == 'DMSE': of,om = _DG_offset_info(degree)
				
				self._offsets[ci].append(of)
				self._offset_mult[ci].append(om)
					
				#compute basis and deriv functions
				if elemname == 'CG': b,d,s = _CG_basis(degree,spts=sptsH1)
				if elemname == 'DG': b,d,s = _DG_basis(degree,spts=sptsL2)
				if elemname == 'CGD': b,d,s = _CGD_basis(degree)
				if elemname == 'DGD': b,d,s = _DGD_basis(degree)
				if elemname == 'CQB': b,d,s = _CQB_basis(degree)
				if elemname == 'DQB': b,d,s = _DQB_basis(degree)
				if elemname == 'CMSE': b,d,s = _CMSE_basis(degree,spts=sptsH1)
				if elemname == 'DMSE': b,d,s = _DMSE_basis(degree,spts=sptsL2)
				self._basis[ci].append(b)
				self._derivs[ci].append(d)
				self._spts[ci].append(s)
			
			#add EmptyElements in
			while len(self._basis[ci]) < 3:
				of,om = _DG_offset_info(0)
				b,d,s = _DG_basis(0)
				self._offsets[ci].append(of)
				self._offset_mult[ci].append(om)
				self._basis[ci].append(b)
				self._derivs[ci].append(d)
				self._spts[ci].append(s)
	
	def get_continuity(self,ci,direc):
		return self._contlist[ci][direc]
	
	def get_nx(self,ci,direc,ncell,bc):
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
		
	def get_nbasis(self,ci,direc):
		return self._nbasis[ci][direc]
		
	def get_ndofs_per_element(self,ci,direc):
		return self._ndofs_per_element[ci][direc]
	
	def get_local_size(self):
		bprod = np.prod(self._nbasis)
		return [bprod,self._ndofs]

	def get_offsets(self,ci,direc):
		return self._offsets[ci][direc],self._offset_mult[ci][direc]
	
	def get_sym_basis(self,ci,direc,newsymb):
		xsymb = sympy.var('x')
		blist = []
		for basis in self._basis[ci][direc]:
			adjbasis = basis.subs(xsymb,newsymb)
			blist.append(adjbasis)
		return blist
		
	def get_sym_derivs(self,ci,direc,newsymb):
		xsymb = sympy.var('x')
		dlist = []
		for deriv in self._derivs[ci][direc]:
			adjderiv = deriv.subs(xsymb,newsymb)
			dlist.append(adjderiv)
		return dlist
	
	def get_basis_exact(self,ci,direc,x):
		
		xsymb = sympy.var('x')
		symbas =  self._basis[ci][direc]
		basis_funcs = sympy.zeros(len(symbas),len(x))
		for i in range(len(symbas)):
			for j in range(len(x)):
				if symbas[i] == 0.5: #check for the constant basis function
					basis_funcs[i,j] = sympy.Rational(1,2)
				else:
					basis_funcs[i,j] = symbas[i].subs(xsymb,x[j])
		return basis_funcs
		
	def get_derivs_exact(self,ci,direc,x):
		xsymb = sympy.var('x')
		symbas =  self._derivs[ci][direc]
		basis_derivs = sympy.zeros(len(symbas),len(x))
		for i in range(len(symbas)):
			for j in range(len(x)):
				if symbas[i] == 0: #check for the constant basis function
					basis_derivs[i,j] = 0
				else:
					basis_derivs[i,j] = symbas[i].subs(xsymb,x[j])
		return basis_derivs
		
	def get_basis(self,ci,direc,x):
		xsymb = sympy.var('x')
		symbas =  self._basis[ci][direc]
		basis_funcs = np.zeros((len(symbas),x.shape[0]))
		for i in range(len(symbas)):
			if symbas[i] == 0.5: #check for the constant basis function
				basis_funcs[i,:] = 0.5
			else:
				basis = sympy.lambdify(xsymb, symbas[i], "numpy")
				basis_funcs[i,:] = np.squeeze(basis(x))
		return basis_funcs

	def get_derivs(self,ci,direc,x):
		xsymb = sympy.var('x')
		symbas =  self._derivs[ci][direc]
		basis_derivs = np.zeros((len(symbas),x.shape[0]))
		for i in range(len(symbas)):
			if symbas[i] == 0.0: #check for the constant basis function
				basis_derivs[i,:] = 0.0
			else:
				deriv = sympy.lambdify(xsymb, symbas[i], "numpy")
				basis_derivs[i,:] = np.squeeze(deriv(x))
		return basis_derivs

		
	def get_icells(self,ci,direc,ncell,bc,interior_facet):
		elemname = self._elemnamelist[ci][direc]
		degree = self._degreelist[ci][direc]
		if elemname == 'CG': return _CG_interaction_cells(ncell,bc,interior_facet,degree)
		if elemname == 'DG': return _DG_interaction_cells(ncell,bc,interior_facet,degree)
		if elemname == 'CGD': return _CGD_interaction_cells(ncell,bc,interior_facet,degree)
		if elemname == 'DGD': return _DGD_interaction_cells(ncell,bc,interior_facet,degree)
		if elemname == 'CQB': return _CG_interaction_cells(ncell,bc,interior_facet,degree) #MIGHT BE WRONG?
		if elemname == 'DQB': return _DG_interaction_cells(ncell,bc,interior_facet,degree) #MIGHT BE WRONG?
		if elemname == 'CMSE': return _CG_interaction_cells(ncell,bc,interior_facet,degree)
		if elemname == 'DMSE': return _DG_interaction_cells(ncell,bc,interior_facet,degree)
				
	def maxdegree(self):
		return max(max(self._degreelist))

	def get_info(self,ci,direc,x):
		b = self.get_basis(ci,direc,x)
		d = self.get_derivs(ci,direc,x)
		return b,d,self._offsets[ci][direc],self._offset_mult[ci][direc]
	
	def get_ncomp(self):
		return self._ncomp

##########  LAGRANGE ELEMENTS  ##########

def _CG_basis(order,spts=None):
	xsymb = sympy.var('x')
	if spts == None:
		spts = gauss_lobatto(order+1)
	symbas = []
	derivs = []
	for i in range(order+1):
		symbas.append(lagrange_poly_support(i,spts,xsymb))
		derivs.append(sympy.diff(symbas[i]))
	
	return symbas,derivs,spts
				
def _DG_basis(order,spts=None):
	xsymb = sympy.var('x')
	symbas = []
	derivs = []
	if order >= 1:
		if spts == None:
			spts = gauss_lobatto(order+1)
		for i in range(order+1):
			symbas.append(lagrange_poly_support(i,spts,xsymb))
			derivs.append(sympy.diff(symbas[i]))
	else:
		spts = [0,]
		symbas.append(sympy.Rational(1,2))
		derivs.append(sympy.Rational(1,1) * 0)	
	
	return symbas,derivs,spts

		
def _CG_offset_info(order):
	offsets = np.arange(0,order+1,dtype=np.int32)
	offset_multiplier = order * np.ones(offsets.shape,dtype=np.int32)
	#'p*i','p*i+1'...'p*i+p'
	return offsets,offset_multiplier

def _DG_offset_info(order):
	offsets = np.arange(0,order+1,dtype=np.int32)
	offset_multiplier = (order+1) * np.ones(offsets.shape,dtype=np.int32)
	#'p*i','p*i+1'...'p*i+p'
	return offsets,offset_multiplier
					
def _CG_interaction_cells(ncell,bc,interior_facet,order):
	if bc == 'periodic':
		off = 0
	else:
		off = 1
	interaction_cells = np.zeros((ncell*order + off,2),dtype=np.int32)
	ilist = np.arange(0,interaction_cells.shape[0])
	rightmost_bound = np.floor_divide(ilist,order) #floor(n/p)
	leftmost_bound = np.floor_divide(ilist-1,order) #floor((n-1)/p)
	#no circular wrapping since we need the actual value in order to get differences in pre-alloc correct ie 2 - (-1) = 3
	if not interior_facet:
		interaction_cells[:,0] = leftmost_bound 
		interaction_cells[:,1] = rightmost_bound 
		if bc == 'nonperiodic':
			interaction_cells[0,0] = interaction_cells[0,1]
			interaction_cells[-1,1] = interaction_cells[-1,0]
	if interior_facet:
		interaction_cells[:,0] = leftmost_bound -1
		interaction_cells[:,1] = rightmost_bound +1
	return interaction_cells

def _DG_interaction_cells(ncell,bc,interior_facet,order):
	interaction_cells = np.zeros((ncell*(order+1),2),dtype=np.int32) 
	ilist = np.arange(0,interaction_cells.shape[0])
	rightmost_bound = np.floor_divide(ilist,order+1) #floor(n/p)
	if not interior_facet:
		interaction_cells[:,0] = rightmost_bound
		interaction_cells[:,1] = rightmost_bound
	if interior_facet:
		interaction_cells[:,0] = rightmost_bound - 1
		interaction_cells[:,1] = rightmost_bound + 1
	return interaction_cells


###########  MGD ELEMENTS   #########


def _CGD_basis(order):
	xsymb = sympy.var('x')
	symbas = []
	derivs = []
	a = 2
	b = -order
	p = a * np.arange(order+1) +b
	spts = [-1,1]
	for i in range(0,order+1):
		basis = lagrange_poly_support(i,p,xsymb)
		symbas.append(basis)
		derivs.append(sympy.diff(basis))
	
	return symbas,derivs,spts

def _DGD_basis(order):
	xsymb = sympy.var('x')
	symbas = []
	derivs = []
	cg_symbas = []

	a = 2
	b = -(order+1)
	p = a * np.arange(order+2) + b
	for i in range(0,order+2):
		cg_symbas.append(lagrange_poly_support(i,p,xsymb))
	
	symbas.append(sympy.diff(-cg_symbas[0]))
	derivs.append(sympy.diff(symbas[0]))
	for i in range(1,order+1):
		dN = sympy.diff(cg_symbas[i])
		basis = symbas[i-1] - dN
		symbas.append(basis)
		derivs.append(sympy.diff(basis))
	spts = [0,]

	return symbas,derivs,spts
	
def _CGD_offset_info(order):
	offsets = np.arange(-(order-1)/2,(order-1)/2+2,1,dtype=np.int32)
	offset_multiplier = np.ones(offsets.shape,dtype=np.int32)
	return offsets,offset_multiplier

def _DGD_offset_info(order):
	offsets = np.arange(-(order)/2,(order)/2+1,1,dtype=np.int32) #THIS MIGHT BE WRONG!
	offset_multiplier = np.ones(offsets.shape,dtype=np.int32)
	return offsets,offset_multiplier

#FIX THIS
#ADD FACET INTEGRAL STUFF
def _CGD_interaction_cells(ncell,bc,interior_facet,order):
	pass

#FIX THIS
#ADD FACET INTEGRAL STUFF	
def _DGD_interaction_cells(ncell,bc,interior_facet,order):
	pass

##GD

	#def get_interaction_cells(self,nxcell,bc,facet_integral):	
		#if bc == 'periodic':
			#off = 0
		#else:
			#off = 1		
		#interaction_offsets = np.array([-(self.order-1)/2-1,(self.order-1)/2],dtype=np.int32)
##		interaction_offsets = np.array([-2,1],dtype=np.int32)
		#interaction_cells = np.zeros((nxcell+off,2),dtype=np.int32) 
		#ilist = np.arange(0,interaction_cells.shape[0])
		#interaction_cells[:,:] = np.expand_dims(ilist,axis=1) + np.expand_dims(interaction_offsets,axis=0)
		##print interaction_cells
		#return interaction_cells
		
##DGD


	#def get_interaction_cells(self,nxcell,bc,facet_integral):	
		#interaction_offsets = np.array([-(self.order-1)/2,(self.order-1)/2],dtype=np.int32)
		#interaction_cells = np.zeros((nxcell,2),dtype=np.int32)
		#ilist = np.arange(0,interaction_cells.shape[0])
		#interaction_cells[:,:] = np.expand_dims(ilist,axis=1) + np.expand_dims(interaction_offsets,axis=0)
		#return interaction_cells
		
###########  QB ELEMENTS   #########


def _CQB_basis(order):
	pass

def _DQB_basis(order):
	pass

#I THINK THESE ARE THE SAME AS CG/DG...BUT MAYBE INTERACTION CELLS IS DIFFERENT?
def _CQB_offset_info(order):
	pass

def _DQB_offset_info(order):
	pass

def _CQB_interaction_cells(ncell,bc,interior_facet,order):
	pass
	
def _DQB_interaction_cells(ncell,bc,interior_facet,order):
	pass

###########  MSE ELEMENTS   #########

_CMSE_basis = _CG_basis

def _DMSE_basis(order,spts=None):
	pass
