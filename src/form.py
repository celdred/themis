
from petscshim import PETSc
import numpy as np
from assembly import two_form_preallocate_opt,AssembleTwoForm,AssembleOneForm,AssembleZeroForm
from tsfc_interface import compile_form

#############3

def create_matrix(mat_type,target,source,blocklist,kernellist):
	
	#block matrix
	if mat_type == 'nest' and (target.nspaces > 1 or source.nspaces > 1):
		#create matrix array
		matrices = []			
		for si1 in xrange(target.nspaces):
			matrices.append([])
			for si2 in xrange(source.nspaces):
				if ((si1,si2) in blocklist):
					bindex = blocklist.index((si1,si2))
					mat = create_mono(target.get_space(si1),source.get_space(si2),[(0,0),],[kernellist[bindex],])
				else:
					mat = create_empty(target.get_space(si1),source.get_space(si2))
				matrices[si1].append(mat)
		
		#do an empty assembly
		for si1 in xrange(target.nspaces):
			for si2 in xrange(source.nspaces):
				if ((si1,si2) in blocklist):
					bindex = blocklist.index((si1,si2))
					fill_mono(matrices[si1][si2],target.get_space(si1),source.get_space(si2),[(0,0),],[kernellist[bindex],],zeroassembly=True)
					#this catches bugs in pre-allocation and the initial assembly by locking the non-zero structure
					matrices[si1][si2].setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
					matrices[si1][si2].setOption(PETSc.Mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)    
					#These are for zeroRows- the first keeps the non-zero structure when zeroing rows, the 2nd tells PETSc that the process only zeros owned rows
					matrices[si1][si2].setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True) 
					matrices[si1][si2].setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, False)	
	
		#create nest
		mat = PETSc.Mat().createNest(matrices,comm=PETSc.COMM_WORLD)
	
	#monolithic matrix
	if (mat_type == 'nest' and (target.nspaces == 1 and source.nspaces == 1)) or mat_type == 'aij':
		#create matrix
		mat = create_mono(target,source,blocklist,kernellist)
		#do an empty assembly
		fill_mono(mat,target,source,blocklist,kernellist,zeroassembly=True)
	
	mat.assemble()

	#this catches bugs in pre-allocation and the initial assembly by locking the non-zero structure
	mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
	mat.setOption(PETSc.Mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)    
	mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
	#These are for zeroRows- the first keeps the non-zero structure when zeroing rows, the 2nd tells PETSc that the process only zeros owned rows
	mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True) 
	mat.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, False)	
	return mat
	
def create_empty(target,source):
	#create matrix			
	mlist = []
	nlist = []
	for si1 in xrange(target.nspaces):
		tspace = target.get_space(si1)
		for ci1 in xrange(tspace.ncomp):
			for bi1 in xrange(tspace.npatches):
				m = tspace.get_localndofs(ci1,bi1)
				mlist.append(m)
	for si2 in xrange(source.nspaces):
		sspace = source.get_space(si2)
		for ci2 in xrange(sspace.ncomp):						
			for bi2 in xrange(sspace.npatches):						
				n = sspace.get_localndofs(ci2,bi2)
				nlist.append(n)
	
	M = np.sum(np.array(mlist,dtype=np.int32))
	N = np.sum(np.array(nlist,dtype=np.int32))
	mat = PETSc.Mat()
	mat.create(PETSc.COMM_WORLD)
	mat.setSizes(((M, None),(N,None)))
	mat.setType('aij')
	mat.setLGMap(target.get_overall_lgmap(),cmap=source.get_overall_lgmap())
	mat.setUp()
	mat.assemblyBegin()
	mat.assemblyEnd()
		
	return mat
	
def create_mono(target,source,blocklist,kernellist):
	#create matrix			
	mlist = []
	nlist = []			
	for si1 in xrange(target.nspaces):
		tspace = target.get_space(si1)
		for ci1 in xrange(tspace.ncomp):
			for bi1 in xrange(tspace.npatches):
				m = tspace.get_localndofs(ci1,bi1)
				mlist.append(m)
	for si2 in xrange(source.nspaces):
		sspace = source.get_space(si2)
		for ci2 in xrange(sspace.ncomp):						
			for bi2 in xrange(sspace.npatches):						
				n = sspace.get_localndofs(ci2,bi2)
				nlist.append(n)
				
	M = np.sum(np.array(mlist,dtype=np.int32))
	N = np.sum(np.array(nlist,dtype=np.int32))
	mat = PETSc.Mat()
	mat.create(PETSc.COMM_WORLD)
	mat.setSizes(((M, None),(N,None)))
	mat.setType('aij')
	mat.setLGMap(target.get_overall_lgmap(),cmap=source.get_overall_lgmap())
	
	#preallocate matrix

	#PRE-ALLOCATE IS NOT QUITE PERFECT FOR FACET INTEGRALS- IT WORKS BUT IS TOO MUCH!
	mlist.insert(0,0)
	mlist_adj = np.cumsum(mlist)
	dnnzarr = np.zeros(M,dtype=np.int32)
	onnzarr = np.zeros(M,dtype=np.int32)
	i = 0 #this tracks which row "block" are at
	#This loop order ensures that we fill an entire row in the matrix first
	#Assuming that fields are stored si,ci,bi, which they are!
	for si1 in xrange(target.nspaces):
		tspace = target.get_space(si1)
		for ci1 in xrange(tspace.ncomp):
			#only pre-allocate diagonal blocks, so one loop on bi
			for bi in xrange(tspace.npatches): #here we can assume tspace.mesh = sspace.mesh, and therefore tspace.blocks = sspace.nblocks)
				for si2 in xrange(source.nspaces):
					sspace = source.get_space(si2)
					for ci2 in xrange(sspace.ncomp):
						if ((si1,si2) in blocklist):
							bindex = blocklist.index((si1,si2))
							interior_x,interior_y,interior_z = get_interior_flags(kernellist[bindex])
							dnnz,onnz = two_form_preallocate_opt(tspace.mesh(),tspace,sspace,ci1,ci2,bi,interior_x,interior_y,interior_z)
							dnnz = np.ravel(dnnz)
							onnz = np.ravel(onnz)
							dnnzarr[mlist_adj[i]:mlist_adj[i+1]] = dnnzarr[mlist_adj[i]:mlist_adj[i+1]] + dnnz
							onnzarr[mlist_adj[i]:mlist_adj[i+1]] = onnzarr[mlist_adj[i]:mlist_adj[i+1]] + onnz
				i = i + 1 #increment row block
	mat.setPreallocationNNZ((dnnzarr,onnzarr))
	mat.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
	mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
	mat.setUp()
	mat.zeroEntries()
	
	return mat

#FIX THIS- NEED TO CHECK INTEGRAL TYPES
#REALLY WE SUPPORT CELL, ds,dS,ds_horiz,ds_vert,dS_bottom,dS_top,dS_horiz
def get_interior_flags(kernellist):
	interior_x = False
	interior_y = False
	interior_z = False
	for kernel in kernellist:
		pass
		#print 'interior flags integral type',kernel.integral_type
	return interior_x,interior_y,interior_z
	

def fill_mono(mat,target,source,blocklist,kernellist,zeroassembly=False):
	#print blocklist
	for si1 in xrange(target.nspaces):
		isrow = target.get_field_lis(si1)
		for si2 in xrange(source.nspaces):
			iscol = source.get_field_lis(si2) 
			if (si1,si2) in blocklist:
				submat = get_block(mat,isrow,iscol)
				bindex = blocklist.index((si1,si2))
				for kernel in kernellist[bindex]:
					AssembleTwoForm(submat,target.get_space(si1),source.get_space(si2),kernel,zeroassembly=zeroassembly)
				restore_block(isrow,iscol,mat,submat)
	
	mat.assemble()


def get_block(matrix,isrow,iscol):
	if isrow == None or iscol == None:
		return matrix
	else:
		return matrix.getLocalSubMatrix(isrow,iscol)

def restore_block(isrow,iscol,matrix,submat):
	if isrow == None or iscol == None:
		pass
	else:
		matrix.restoreLocalSubMatrix(isrow,iscol,submat)
		
########################
	
class OneForm():
	
	def __init__(self,F,activefield,bcs=[],pre_f_callback=None):
		self.F = F
		self.bcs = bcs
		self._pre_f_callback = pre_f_callback
		
		self.space = self.F.arguments()[0].function_space()
		#NAMES?

		self.activefield = activefield

		#create vector
		self.vector = self.space.get_composite_da().createGlobalVec()
		self.vector.set(0.0)
		self.lvectors = []
		for si in xrange(self.space.nspaces):
			for ci in xrange(self.space.get_space(si).ncomp):
				for bi in xrange(self.space.get_space(si).npatches):
					self.lvectors.append(self.space.get_space(si).get_da(ci,bi).createLocalVector())
		
		#compile local assembly kernels
		idx_kernels = compile_form(F)
		
		self.local_assembly_kernels = {}
		for idx,subkernels in idx_kernels:
			self.local_assembly_kernels[idx] = subkernels

	#BROKEN- NAMES STUFF...
	def output(self,view,ts=None):
		with self.space.composite_da.getAccess(self.vector) as splitglobalvec:
			k = 0 #this gives the index into the list of names
			for si in xrange(self.space.nspaces):
				soff = self.space.get_space_offset(si)
				for ci in xrange(self.space.get_space(si).ncomp):
					coff = self.space.get_space(si).get_component_offset(ci)
					for bi in xrange(self.space.get_space(si).nblocks):
						if not (ts == None):
							tname = self.names[k] + str(bi) + '_' + str(ts)
						else:
							tname = self.names[k] + str(bi)
						vector_output(splitglobalvec[bi+soff+coff],tname,view)
					k = k + 1
	
	def assembleform(self,snes,X,F):

		#"copy" X (petsc Vec) into "active" field
		self.activefield._activevector = X
			
		#pre-function callback
		if not (self._pre_f_callback == None): self._pre_f_callback(X)
				
		#assemble
		self._assemblehelper(F)
		
		#restore the old active field
		self.activefield._activevector = self.activefield._vector
			
		#assemble X and F
		X.assemble()
		F.assemble()

	def _assemblehelper(self,vec):
			
		#zero out vector
		vec.set(0.0)
		for lvec in self.lvectors:
			lvec.set(0.0)
			
		#assemble
		blocklist = self.local_assembly_kernels.keys()
		kernellist = self.local_assembly_kernels.values()
		for si1 in xrange(self.space.nspaces):
			soff = self.space.get_space_offset(si1)
			if (si1,) in blocklist:
				bindex = blocklist.index((si1,))
				for kernel in kernellist[bindex]:
					AssembleOneForm(self.lvectors[soff:soff+self.space.get_space(si1).ncomp*self.space.get_space(si1).npatches],self.space.get_space(si1),kernel)
		
		self.space.get_composite_da().gather(vec,PETSc.InsertMode.ADD_VALUES,self.lvectors)
		
		#Apply symmetric essential boundaries		
		#set residual to ZERO at boundary points, since we set x(boundary) = bvals in non-linear solve and also apply boundary conditions to Jacobian
		for bc in self.bcs:
			bc.apply_vector(vec,zero=True)

	def destroy(self):
		self.vector.destroy()
		for v in self.lvectors:
			v.destroy()
			
class TwoForm():
	def __init__(self,J,activefield,Jp=None,mat_type='aij',pmat_type='aij',constantJ=False,constantP=False,bcs=[],pre_j_callback=None):
		
		self.target = J.arguments()[0].function_space()
		self.source = J.arguments()[1].function_space()
		
		self.activefield = activefield
		self._pre_j_callback = pre_j_callback

		self.bcs = bcs
		self.Jconstant = constantJ
		self.Pconstant = constantP
		self.Jassembled = False
		self.Passembled = False
		if self.Jconstant: assert (self.Pconstant == True)
		
		self.mat_type = mat_type
		self.pmat_type = pmat_type
		self.J = J
		self.Jp = Jp

		#compile local assembly kernels
		idx_kernels = compile_form(J)
		self.mat_local_assembly_kernels = {}
		for idx,subkernels in idx_kernels:
			self.mat_local_assembly_kernels[idx] = subkernels
		
		if not self.Jp == None:
			idx_kernels = compile_form(self.Jp)
			self.pmat_local_assembly_kernels = {}
			for idx,subkernels in idx_kernels:
				self.pmat_local_assembly_kernels[idx] = subkernels
		
		#create matrices
		self.mat = create_matrix(mat_type,self.target,self.source,self.mat_local_assembly_kernels.keys(),self.mat_local_assembly_kernels.values())
		if not self.Jp == None:
			self.pmat = create_matrix(pmat_type,self.target,self.source,self.pmat_local_assembly_kernels.keys(),self.pmat_local_assembly_kernels.values())
			
	def destroy(self):
		self.mat.destroy() #DOES DESTROYING A NEST AUTOMATICALLY DESTROY THE SUB MATRICES?
		if not self.Jp == None:
			self.pmat.destroy()

	def assembleform(self,snes,X,J,P):

		#"copy" X into "active" field
		self.activefield._activevector = X
		
		if (not ((self.Jconstant == True) and (self.Jassembled == True))) or ((not (self.Jp == None)) and (not ((self.Pconstant == True) and (self.Passembled == True)))):
			#pre-jacobian callback
			if not (self._pre_j_callback == None): self._pre_j_callback(X)
			
		if not ((self.Jconstant == True) and (self.Jassembled == True)):
			
			#assemble
			self._assemblehelper(J,self.mat_type,self.mat_local_assembly_kernels.keys(),self.mat_local_assembly_kernels.values())
			self.Jassembled = True		

		if (not (self.Jp == None)) and (not ((self.Pconstant == True) and (self.Passembled == True))):

			#assemble
			self._assemblehelper(P,self.pmat_type,self.pmat_local_assembly_kernels.keys(),self.pmat_local_assembly_kernels.values())
			self.Passembled = True

		#restore the old active field
		self.activefield._activevector = self.activefield._vector

		#WHAT SHOULD I REALLY BE RETURNING HERE?
		return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

	def _assemblehelper(self,mat,mat_type,blocklist,kernellist):
		
		#zero out matrix
		mat.zeroEntries()
		
		#assemble
		fill_mono(mat,self.target,self.source,blocklist,kernellist)
				
		#Apply symmetric essential boundaries
		for bc in self.bcs:
			bc.apply_matrix(mat,mat_type)

class ZeroForm():
	
	def __init__(self,E):
		self.E = E
		idx_kernels = compile_form(E)
		self.kernellist = idx_kernels[0][1] #This works because we only have 1 idx (0-forms have no arguments) and therefore only 1 kernel list
		self.value = 0.
		self.mesh = self.E.ufl_domain()

	def assembleform(self):
		return AssembleZeroForm(self.mesh,self.kernellist)
		
