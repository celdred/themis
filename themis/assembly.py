import numpy as np
from petscshim import PETSc
from codegenerator import generate_assembly_routine
import instant
from compilation_options import *
from tsfc_interface import compile_form
from ufl import VectorElement
from mpi4py import MPI
from function import Function
from constant import Constant
import functools

def compile_functional(kernel,tspace,sspace,mesh):
	
	#NEEDS FIXING FOR CONSTANTS
	if not kernel.zero:

		#### THIS IS UGLY HACKING TO FIX INABILITY TO GET KERNELS TO WORK WITH CONST * CONST * RESTRICT STUFF ####
		### REMOVE ONCE I FIGURE OUT WHAT IS GOING ON ########
		ncoeff = len(kernel.coefficient_numbers)
		operands = kernel.ast.operands()[0]
		func_args = operands[2]

		elem = mesh.coordinates.function_space().themis_element()
		#first argument is the local tensor and second is always coords...
		func_args[1].qual = []
		func_args[1].pointers = []
		func_args[1].sym.rank = elem.get_local_size()
		
		k = 0
		for i in range(ncoeff):
			field = kernel.coefficients[kernel.coefficient_map[i]]
			if isinstance(field,Function):
				for si in range(field.function_space().nspaces):
					elem = field.function_space().get_space(si).themis_element()
					arg = func_args[2+k]
					arg.qual = []
					arg.pointers = []
					arg.sym.rank = elem.get_local_size()
					k = k + 1
			if isinstance(field,Constant):
				arg = func_args[2+k]
				arg.qual = []
				arg.pointers = []
				if len(field.dat.shape) >= 1: arg.sym.rank = np.prod(field.dat.shape)
				if len(field.dat.shape) == 0: arg.sym.rank = (1,)
				k = k + 1
				
	###############################
	
	#THIS NEEDS SOME SORT OF CACHING CHECK!
	assembly_routine = generate_assembly_routine(mesh,tspace,sspace,kernel)
	#assembly_routine = assembly_routine.encode('ascii','ignore')
	kernel.assemble_function= instant.build_module( code=assembly_routine,
		  include_dirs=include_dirs,
		  library_dirs=library_dirs,
		  libraries=libraries,
		  init_code = '    import_array();',
		  cppargs=['-O3',],
		  swig_include_dirs=swig_include_dirs).assemble

	if not kernel.zero:
		kernel.assemblycompiled = True

#BROKEN FOR MULTIPATCH	
def extract_fields(kernel):

	#EVENTUALLY SPLIT OUT CREATION OF ARGS LIST FROM SCATTERING!
	
	#create field args list: matches construction done in codegenerator.py
	fieldargs_list = []
	constantargs_list = []

	if not kernel.zero:

		#append coordinates
		fieldargs_list.append(kernel.mesh.coordinates.function_space().get_da(0,0)) #THIS BREAKS FOR MULTIPATCH MESHES
		fieldargs_list.append(kernel.mesh.coordinates._lvectors[0]) #THIS BREAKS FOR MULTIPATCH MESHES
		
		#THIS NEEDS FIXING FOR CONSTANTS...
		for fieldindex in kernel.coefficient_map:

			field = kernel.coefficients[fieldindex]
			if isinstance(field,Function):
				field.scatter()
				for si in range(field.function_space().nspaces):
					fspace = field.function_space().get_space(si)
					bi = 0 #BROKEN FOR MULTIPATCH
					for ci in range(fspace.ncomp):
						fieldargs_list.append(fspace.get_da(ci,bi))
						fieldargs_list.append(field.get_lvec(si,ci,bi))
			if isinstance(field,Constant):
				data = field.dat
				constantargs_list.append(float(data))
				#BROKEN FOR VECTOR/TENSOR CONSTANTS

	return fieldargs_list + constantargs_list
	
def AssembleTwoForm(mat,tspace,sspace,kernel,zeroassembly=False):

	#FIX THIS- SHOULD BE TIED TO KERNEL!
	name = 'test'
	
	with PETSc.Log.Stage(name + '_assemble'):
		mesh = tspace.mesh()

		if zeroassembly:
			kernel.zero = True
		#compile functional IFF form not already compiled
		#also extract coefficient and geometry args lists
		with PETSc.Log.Event('compile'):
			if not kernel.assemblycompiled:
				compile_functional(kernel,tspace,sspace,mesh)
 
		#scatter fields into local vecs
		with PETSc.Log.Event('extract'):
			fieldargs_list = extract_fields(kernel)

		#assemble
		with PETSc.Log.Event('assemble'):
			
			#THIS STUFF IS BROKEN FOR MULTIPATCH MESHES
			#SHOULD REALLY DO 1 ASSEMBLY PER PATCH
			#get the list of das
			tdalist = []
			for ci1 in range(tspace.ncomp):
				for bi in range(mesh.npatches):
					tdalist.append(tspace.get_da(ci1,bi))
			sdalist = []
			for ci2 in range(sspace.ncomp):
				for bi in range(mesh.npatches):
					sdalist.append(sspace.get_da(ci2,bi))

			#get the block sub matrices
			submatlist = []
			for ci1 in range(tspace.ncomp):
				for ci2 in range(sspace.ncomp):
					for bi in range(mesh.npatches):
						isrow_block = tspace.get_component_block_lis(ci1,bi)
						iscol_block = sspace.get_component_block_lis(ci2,bi)
						submatlist.append(mat.getLocalSubMatrix(isrow_block,iscol_block))
			
			if kernel.integral_type == 'cell':
				da = mesh.get_cell_da(0)
			if kernel.integral_type in ['interior_facet_x','exterior_facet_x_top','exterior_facet_x_bottom']:
				da = mesh._edgex_das[0]
			if kernel.integral_type in ['interior_facet_y','exterior_facet_y_top','exterior_facet_y_bottom']:
				da = mesh._edgey_das[0]
			if kernel.integral_type in ['interior_facet_z','exterior_facet_z_top','exterior_facet_z_bottom']:
				da = mesh._edgez_das[0]
				
				#BROKEN FOR MULTIPATCH- FIELD ARGS LIST NEEDS A BI INDEX
			kernel.assemble_function(da,*(submatlist + tdalist + sdalist + fieldargs_list))
			
			#restore sub matrices
			k=0
			for ci1 in range(tspace.ncomp):
					for ci2 in range(sspace.ncomp):
						for bi in range(mesh.npatches):
							isrow_block = tspace.get_component_block_lis(ci1,bi)
							iscol_block = sspace.get_component_block_lis(ci2,bi)
							mat.restoreLocalSubMatrix(isrow_block,iscol_block,submatlist[k])
							k=k+1

		if zeroassembly:
			kernel.zero = False

def AssembleZeroForm(mesh,kernellist):
	
	#FIX THIS NAME
	name = 'test'
	
	with PETSc.Log.Stage(name + '_assemble'):

		value = 0.

		for kernel in kernellist:
		#compile functional IFF form not already compiled
		#also extract coefficient and geometry args lists
			with PETSc.Log.Event('compile'):
				if not kernel.assemblycompiled:
					compile_functional(kernel,None,None,mesh)
 
		#scatter fields into local vecs
			with PETSc.Log.Event('extract'):
				fieldargs_list = extract_fields(kernel)

			#assemble the form
			with PETSc.Log.Event('assemble'):
				#BROKEN FOR MULTIPATCH MESHES
				#SHOULD DO 1 ASSEMBLY PER PATCH

				if kernel.integral_type == 'cell':
					da = mesh.get_cell_da(0)
				if kernel.integral_type in ['interior_facet_x','exterior_facet_x_top','exterior_facet_x_bottom']:
					da = mesh._edgex_das[0]
				if kernel.integral_type in ['interior_facet_y','exterior_facet_y_top','exterior_facet_y_bottom']:
					da = mesh._edgey_das[0]
				if kernel.integral_type in ['interior_facet_z','exterior_facet_z_top','exterior_facet_z_bottom']:
					da = mesh._edgez_das[0]

					#BROKEN FOR MULTIPATCH- FIELD ARGS LIST NEEDS A BI INDEX
				lvalue = kernel.assemble_function(da,*(fieldargs_list))

				if PETSc.COMM_WORLD.Get_size() == 1:
					value = value + lvalue
				else:
					tbuf = np.array(0,'d')
					mpicomm = PETSc.COMM_WORLD.tompi4py()
					mpicomm.Allreduce([np.array(lvalue,'d'),MPI.DOUBLE],[tbuf,MPI.DOUBLE],op=MPI.SUM) #this defaults to sum, so we are good
					value = value + np.copy(tbuf)

	return value

def AssembleOneForm(veclist,space,kernel):
	
	#print('oneform',kernel.integral_type)
	
	name = 'test'
	
	with PETSc.Log.Stage(name + '_assemble'):

		mesh = space.mesh()
			
		#compile functional IFF form not already compiled
		#also extract coefficient and geometry args lists
		with PETSc.Log.Event('compile'):
			if not kernel.assemblycompiled:
				compile_functional(kernel,space,None,mesh)
				
		#scatter fields into local vecs
		with PETSc.Log.Event('extract'):
			fieldargs_list = extract_fields(kernel)
		
		#assemble
		with PETSc.Log.Event('assemble'):
			#BROKEN FOR MULTIPATCH MESHES
			#SHOULD DO 1 ASSEMBLY PER PATCH
			
			if kernel.integral_type == 'cell':
				da = mesh.get_cell_da(0)
			if kernel.integral_type in ['interior_facet_x','exterior_facet_x_top','exterior_facet_x_bottom']:
				da = mesh._edgex_das[0]
			if kernel.integral_type in ['interior_facet_y','exterior_facet_y_top','exterior_facet_y_bottom']:
				da = mesh._edgey_das[0]
			if kernel.integral_type in ['interior_facet_z','exterior_facet_z_top','exterior_facet_z_bottom']:
				da = mesh._edgez_das[0]
				
			#get the list of das
			tdalist = []
			for ci1 in range(space.ncomp):
				for bi in range(mesh.npatches):
					tdalist.append(space.get_da(ci1,bi))

				#BROKEN FOR MULTIPATCH- FIELD ARGS LIST NEEDS A BI INDEX
			kernel.assemble_function(da,*(veclist + tdalist + fieldargs_list))
			
def compute_1d_bounds(ci1,ci2,i,elem1,elem2,ncell,ndofs,interior_facet,bc,ranges1,ranges2):
	dnnz = np.zeros((ndofs),dtype=np.int32) 
	nnz = np.zeros((ndofs),dtype=np.int32)

	#clip values to a range
	py_clip = lambda x, l, u: l if x < l else u if x > u else x
	
	icells = elem1.get_icells(ci1,i,ncell,bc,interior_facet)
	leftmost_offsets,leftmost_offsetmult = elem2.get_offsets(ci2,i)
	rightmost_offsets,rightmost_offsetmult = elem2.get_offsets(ci2,i)
	
	leftmostcells = icells[:,0]
	rightmostcells = icells[:,1]

	for i in range(ranges1[0],ranges1[1]):
		leftmostcell = leftmostcells[i]
		rightmostcell = rightmostcells[i]
		leftbound = leftmost_offsets[0] + leftmostcell * leftmost_offsetmult[0]
		rightbound = rightmost_offsets[-1] + rightmostcell * rightmost_offsetmult[-1]
		nnz[i-ranges1[0]] = rightbound - leftbound + 1 #this is the total size
		dnnz[i-ranges1[0]] = py_clip(rightbound,ranges2[0],ranges2[1]-1) - py_clip(leftbound,ranges2[0],ranges2[1]-1) + 1
	return dnnz,nnz


#This computes the preallocation between component ci1, block bi of space1 and component ci2, block bi of space2
#Where space1 and space2 are FunctionSpace (or the enriched versions)
#and interior_x, etc. indicate that an interior facet integral is happening (needed for facet integral preallocation)
#This is perfect for non-facet integrals in all dimensions, and somewhat overestimates facet integrals for 2D or 3D
	
#THIS IS BROKEN FOR VECTOR SPACES
#NEED TO MULTIPLY BY NDOFS (maybe ndofs^ndim?) I THINK...
#ALSO SIZING ON DNNZ AND NNZ ARRAYS IS BROKEN
def two_form_preallocate_opt(mesh,space1,space2,ci1,ci2,bi,interior_x,interior_y,interior_z):
	
	elem1 = space1.themis_element()
	elem2 = space2.themis_element()

	space1_ranges = space1.get_xy(ci1,bi)
	space2_ranges = np.array(space2.get_xy(ci2,bi),dtype=np.int32)
	nx1s = space1.get_local_nxny(ci1,bi)
	
	swidth = space2.get_da(ci2,bi).getStencilWidth()
	
	xyzmaxs_space2 = space2.get_nxny(ci2,bi)
		
	#compute i bounds
	dnnz_x,nnz_x = compute_1d_bounds(ci1,ci2,0,elem1,elem2,mesh.nxs[bi][0],nx1s[0],interior_x,mesh.bcs[0],space1_ranges[0],space2_ranges[0])
	
	#compute j bounds	
	if mesh.ndim >= 2:
		dnnz_y,nnz_y = compute_1d_bounds(ci1,ci2,1,elem1,elem2,mesh.nxs[bi][1],nx1s[1],interior_y,mesh.bcs[1],space1_ranges[1],space2_ranges[1])

	#compute k bounds
	if mesh.ndim == 3:
		dnnz_z,nnz_z = compute_1d_bounds(ci1,ci2,2,elem1,elem2,mesh.nxs[bi][2],nx1s[2],interior_z,mesh.bcs[2],space1_ranges[2],space2_ranges[2])

	#fix periodic boundaries when there is only 1 processor in that direction
	nprocs = mesh._cell_das[bi].getProcSizes()
	#for bc,nproc in zip(mesh.bcs,nprocs):
	if (mesh.bcs[0] == 'periodic') and (nprocs[0] == 1):
		dnnz_x = nnz_x
	if mesh.ndim >=2:
		if (mesh.bcs[1] == 'periodic') and (nprocs[1] == 1):
			dnnz_y = nnz_y
	if mesh.ndim == 3:
		if (mesh.bcs[2] == 'periodic') and (nprocs[2] == 1):
			dnnz_z = nnz_z
	
	if mesh.ndim == 1:
		dnnzarr = dnnz_x
		nnzarr = nnz_x
	if mesh.ndim == 2:
		#dnnzarr = reduce(np.multiply, np.ix_(dnnz_y,dnnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
		#nnzarr = reduce(np.multiply, np.ix_(nnz_y,nnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
		dnnzarr = functools.reduce(np.multiply, np.ix_(dnnz_y,dnnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
		nnzarr = functools.reduce(np.multiply, np.ix_(nnz_y,nnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
	if mesh.ndim == 3:
		#dnnzarr = reduce(np.multiply, np.ix_(dnnz_z,dnnz_y,dnnz_x))
		#nnzarr = reduce(np.multiply, np.ix_(nnz_z,nnz_y,nnz_x))
		dnnzarr = functools.reduce(np.multiply, np.ix_(dnnz_z,dnnz_y,dnnz_x))
		nnzarr = functools.reduce(np.multiply, np.ix_(nnz_z,nnz_y,nnz_x))
	onnzarr = nnzarr - dnnzarr
	
		#see http://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors

	return dnnzarr,onnzarr


#this is a helper function to create a TwoForm, given a UFL Form
#Mostly intended for applications that want to use Themis/UFL to handle the creation and assembly of a PETSc matrix, and then do something with it
#Doesn't interact with solver stuff, although this might be subject to change at some point

#IS THIS GOOD NOW?

def assemble(f,bcs=None,form_compiler_parameters=None,mat_type='aij'):
	import ufl
	from solver import _extract_bcs
	from form import TwoForm

	if not isinstance(f, ufl.Form):
		raise TypeError("Provided 2-Form is a '%s', not a Form" % type(f).__name__)

	if len(f.arguments()) != 2:
		raise ValueError("Provided 2-Form is not a bilinear form")

	bcs = _extract_bcs(bcs)
	
	#FIX HOW FORM COMPILER PARAMETERS ARE HANDLED

	form = TwoForm(f,None,mat_type=mat_type,bcs=bcs,constantJ=True,constantP=True)
	form._assemblehelper(form.mat,form.mat_type,form.mat_local_assembly_kernels.keys(),form.mat_local_assembly_kernels.values())
	form.Jassembled = True

	return form
	
