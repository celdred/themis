import numpy as np
from petscshim import PETSc
from codegenerator import generate_assembly_routine,generate_evaluate_routine
# import instant
# from compilation_options import *
# from tsfc_interface import compile_form
# from ufl import VectorElement
from mpi4py import MPI
from function import Function
from constant import Constant
import functools
#import time

from pyop2.utils import get_petsc_dir
from pyop2 import compilation
import ctypes


class CompiledKernel(object):
    base_cppargs = ["-I%s/include" % d for d in get_petsc_dir()]
    base_ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + ["-lpetsc", "-lm"]

    def __init__(self, source, function_name, restype=ctypes.c_int, cppargs=None, argtypes=None, comm=None):
        if cppargs is None:
            cppargs = self.base_cppargs
        else:
            cppargs = cppargs + self.base_cppargs

        funptr = compilation.load(source, "c", function_name,
                                  cppargs=cppargs,
                                  ldargs=self.base_ldargs,
                                  restype=restype,
                                  argtypes=argtypes,
                                  comm=comm)

        self.funptr = funptr

    def __call__(self, petsc_args, constant_args):
        args = [p.handle for p in petsc_args] + constant_args
        return self.funptr(*args)


def compile_functional(kernel, tspace, sspace, mesh):

    # create da list
    kernel.dalist = []
    facet_direc_list = []
    facet_exterior_boundary_list = []

    if kernel.integral_type == 'cell':
        kernel.dalist.append(mesh.get_cell_da())
        facet_direc_list.append('')
        facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'interior_facet':
        kernel.dalist.append(mesh._edgex_da)
        facet_direc_list.append(0)
        facet_exterior_boundary_list.append('')
        if mesh.ndim >= 2:
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')
        if mesh.ndim >= 3:
            kernel.dalist.append(mesh._edgez_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'exterior_facet':
        if mesh.bcs[0] == 'nonperiodic':
            kernel.dalist.append(mesh._edgex_da)
            kernel.dalist.append(mesh._edgex_da)
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('lower')
        if mesh.ndim >= 2 and mesh.bcs[1] == 'nonperiodic':
            kernel.dalist.append(mesh._edgey_da)
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')
        if mesh.ndim >= 3 and mesh.bcs[2] == 'nonperiodic':
            kernel.dalist.append(mesh._edgez_da)
            kernel.dalist.append(mesh._edgez_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('lower')

    if kernel.integral_type == 'interior_facet_vert': # side facets
        kernel.dalist.append(mesh._edgex_da)
        facet_direc_list.append(0)
        facet_exterior_boundary_list.append('')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')
            
    if kernel.integral_type == 'interior_facet_horiz': # extruded facets
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh._edgez_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('')
            
    if kernel.integral_type == 'exterior_facet_vert': # side exterior facets
        if mesh.bcs[0] == 'nonperiodic':
            kernel.dalist.append(mesh._edgex_da)
            kernel.dalist.append(mesh._edgex_da)
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('lower')
        if mesh.extrusion_dim == 2 and mesh.bcs[1] == 'nonperiodic':
            kernel.dalist.append(mesh._edgey_da)
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')
            
    if kernel.integral_type == 'exterior_facet_top': # extruded exterior facet top
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh._edgez_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('upper')
                 
    if kernel.integral_type == 'exterior_facet_bottom': # extruded exterior facet bottom
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh._edgey_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh._edgez_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('lower')

    # create field args list: matches construction used in codegenerator.py

# MAYBE THIS SHOULD REALLY BE PART OF CODE GENERATOR.PY?

    kernel.fieldargs_list = []
    kernel.constantargs_list = []
    fieldargtypeslist = []
    constantargtypeslist = []

    # append coordinates
    if not kernel.zero:  # don't build any field args stuff for zero kernel
        kernel.fieldargs_list.append(kernel.mesh.coordinates.function_space().get_da(0))
        kernel.fieldargs_list.append(kernel.mesh.coordinates._lvectors[0])
        fieldargtypeslist.append(ctypes.c_voidp)
        fieldargtypeslist.append(ctypes.c_voidp)

        for fieldindex in kernel.coefficient_map:

            field = kernel.coefficients[fieldindex]
            if isinstance(field, Function):
                for si in range(field.function_space().nspaces):
                    fspace = field.function_space().get_space(si)
                    for ci in range(fspace.ncomp):
                        kernel.fieldargs_list.append(fspace.get_da(ci))
                        kernel.fieldargs_list.append(field.get_lvec(si, ci))
                        fieldargtypeslist.append(ctypes.c_voidp)  # DA
                        fieldargtypeslist.append(ctypes.c_voidp)  # Vec
            if isinstance(field, Constant):
                constantargtypeslist.append(ctypes.c_double)

# THIS NEEDS SOME SORT OF CACHING CHECK- BASICALLY WE SHOULDN'T REALLY RE-GENERATE THE BIG C STRING EVERY TIME...
    kernel.assemblyfunc_list = []
    tensorlist = []

    if (tspace is not None) and (sspace is not None):  # 2-form

            for ci1 in range(tspace.ncomp):  # Mat
                for ci2 in range(sspace.ncomp):
                    tensorlist.append(ctypes.c_voidp)
            for ci1 in range(tspace.ncomp):  # tda
                tensorlist.append(ctypes.c_voidp)
            for ci2 in range(sspace.ncomp):  # sda
                tensorlist.append(ctypes.c_voidp)

    if (tspace is not None) and (sspace is None):  # 1-form

            for ci1 in range(tspace.ncomp):  # Vec
                tensorlist.append(ctypes.c_voidp)
            for ci1 in range(tspace.ncomp):  # tda
                tensorlist.append(ctypes.c_voidp)
                    
    if kernel.evaluate == True:
        tensorlist.append(ctypes.c_voidp) # DM
        tensorlist.append(ctypes.c_voidp) # Vec
		
    argtypeslist = [ctypes.c_voidp, ] + tensorlist + fieldargtypeslist + constantargtypeslist

    restype = ctypes.c_int
    if (tspace is None) and (sspace is None):
        restype = ctypes.c_double  # 0-form

    for facet_direc, facet_exterior_boundary in zip(facet_direc_list, facet_exterior_boundary_list):
        kernel.facet_direc = facet_direc
        kernel.facet_exterior_boundary = facet_exterior_boundary

        if kernel.evaluate == True:
            assembly_routine = generate_evaluate_routine(mesh, kernel)
        else:
            assembly_routine = generate_assembly_routine(mesh, tspace, sspace, kernel)
        assembly_function = CompiledKernel(assembly_routine, "assemble", cppargs=["-O3"], argtypes=argtypeslist, restype=restype)
        kernel.assemblyfunc_list.append(assembly_function)

    if not kernel.zero:
        kernel.assemblycompiled = True


def extract_fields(kernel):

    if not kernel.zero:

        kernel.constantargs_list = []

        for fieldindex in kernel.coefficient_map:

            field = kernel.coefficients[fieldindex]
            if isinstance(field, Function):
                field.scatter()

            if isinstance(field, Constant):
                data = field.dat
                kernel.constantargs_list.append(float(data))
                # BROKEN FOR VECTOR/TENSOR CONSTANTS


def AssembleTwoForm(mat, tspace, sspace, kernel, zeroassembly=False):

    # FIX THIS- SHOULD BE TIED TO KERNEL!
    name = 'test'

    with PETSc.Log.Stage(name + '_assemble'):
        mesh = tspace.mesh()

        if zeroassembly:
            kernel.zero = True
        # compile functional IFF form not already compiled
        # also extract coefficient and geometry args lists
        #time1 = time.time()
        with PETSc.Log.Event('compile'):
            if not kernel.assemblycompiled:
                compile_functional(kernel, tspace, sspace, mesh)
        #print('compiled-2',time.time()-time1,zeroassembly)
		
        # scatter fields into local vecs
        #time1 = time.time()
        with PETSc.Log.Event('extract'):
            extract_fields(kernel)
        #print('extracted-2',time.time()-time1,zeroassembly)

        # assemble
        #time1 = time.time()
        with PETSc.Log.Event('assemble'):

            # get the list of das
            tdalist = []
            for ci1 in range(tspace.ncomp):
                tdalist.append(tspace.get_da(ci1))
            sdalist = []
            for ci2 in range(sspace.ncomp):
                sdalist.append(sspace.get_da(ci2))

            # get the block sub matrices
            submatlist = []
            for ci1 in range(tspace.ncomp):
                for ci2 in range(sspace.ncomp):
                    isrow_block = tspace.get_component_block_lis(ci1)
                    iscol_block = sspace.get_component_block_lis(ci2)
                    submatlist.append(mat.getLocalSubMatrix(isrow_block, iscol_block))

            for da, assemblefunc in zip(kernel.dalist, kernel.assemblyfunc_list):
                #PETSc.Sys.Print('assembling 2-form',kernel.integral_type,da)
                #PETSc.Sys.Print(submatlist)
                #PETSc.Sys.Print(tdalist)
                #PETSc.Sys.Print(sdalist)
                #PETSc.Sys.Print(kernel.fieldargs_list)
                #PETSc.Sys.Print(kernel.constantargs_list)
                assemblefunc([da, ] + submatlist + tdalist + sdalist + kernel.fieldargs_list, kernel.constantargs_list)

            # restore sub matrices
            k = 0
            for ci1 in range(tspace.ncomp):
                for ci2 in range(sspace.ncomp):
                    isrow_block = tspace.get_component_block_lis(ci1)
                    iscol_block = sspace.get_component_block_lis(ci2)
                    mat.restoreLocalSubMatrix(isrow_block, iscol_block, submatlist[k])
                    k = k+1

        if zeroassembly:
            kernel.zero = False
        #print('assembled-2',time.time()-time1,zeroassembly)


def AssembleZeroForm(mesh, kernellist):

    # FIX THIS NAME
    name = 'test'

    with PETSc.Log.Stage(name + '_assemble'):

        value = 0.

        for kernel in kernellist:
            # compile functional IFF form not already compiled
            # also extract coefficient and geometry args lists
            with PETSc.Log.Event('compile'):
                if not kernel.assemblycompiled:
                    compile_functional(kernel, None, None, mesh)

        # scatter fields into local vecs
            with PETSc.Log.Event('extract'):
                extract_fields(kernel)

            # assemble the form
            with PETSc.Log.Event('assemble'):

                for da, assemblefunc in zip(kernel.dalist, kernel.assemblyfunc_list):
                    lvalue = assemblefunc([da, ] + kernel.fieldargs_list, kernel.constantargs_list)

                if PETSc.COMM_WORLD.Get_size() == 1:
                    value = value + lvalue
                else:
                    tbuf = np.array(0, 'd')
                    mpicomm = PETSc.COMM_WORLD.tompi4py()
                    mpicomm.Allreduce([np.array(lvalue, 'd'), MPI.DOUBLE], [tbuf, MPI.DOUBLE], op=MPI.SUM)  # this defaults to sum, so we are good
                    value = value + np.copy(tbuf)

    return value


def AssembleOneForm(veclist, space, kernel):

    # FIX THIS NAME
    name = 'test'

    with PETSc.Log.Stage(name + '_assemble'):

        mesh = space.mesh()

        # compile functional IFF form not already compiled
        # also extract coefficient and geometry args lists
        #time1 = time.time()
        with PETSc.Log.Event('compile'):
            if not kernel.assemblycompiled:
                compile_functional(kernel, space, None, mesh)
        #print('compiled-1',time.time()-time1)

        # scatter fields into local vecs
        #time1 = time.time()
        with PETSc.Log.Event('extract'):
            extract_fields(kernel)
        #print('extracted-1',time.time()-time1)

        # assemble
        #time1 = time.time()
        with PETSc.Log.Event('assemble'):

            # get the list of das
            tdalist = []
            for ci1 in range(space.ncomp):
                tdalist.append(space.get_da(ci1))

            for da, assemblefunc in zip(kernel.dalist, kernel.assemblyfunc_list):
                #PETSc.Sys.Print('assembling 1-form',kernel.integral_type,da)
                #PETSc.Sys.Print(veclist)
                #PETSc.Sys.Print(tdalist)
                #PETSc.Sys.Print(kernel.fieldargs_list)
                #PETSc.Sys.Print(kernel.constantargs_list)
                #PETSc.Sys.Print(kernel.fieldargs_list[0].getGhostRanges())
                assemblefunc([da, ] + veclist + tdalist + kernel.fieldargs_list, kernel.constantargs_list)
        #print('assembled-1',time.time()-time1)


def compute_1d_bounds(ci1, ci2, i, elem1, elem2, ncell, ndofs, interior_facet, bc, ranges1, ranges2):
    dnnz = np.zeros((ndofs), dtype=np.int32)
    nnz = np.zeros((ndofs), dtype=np.int32)

    # clip values to a range
    py_clip = lambda x, l, u: l if x < l else u if x > u else x

    icells = elem1.get_icells(ci1, i, ncell, bc, interior_facet)
    leftmost_offsets, leftmost_offsetmult = elem2.get_offsets(ci2, i)
    rightmost_offsets, rightmost_offsetmult = elem2.get_offsets(ci2, i)

    leftmostcells = icells[:, 0]
    rightmostcells = icells[:, 1]

    for i in range(ranges1[0], ranges1[1]):
        leftmostcell = leftmostcells[i]
        rightmostcell = rightmostcells[i]
        leftbound = leftmost_offsets[0] + leftmostcell * leftmost_offsetmult[0]
        rightbound = rightmost_offsets[-1] + rightmostcell * rightmost_offsetmult[-1]
        nnz[i-ranges1[0]] = rightbound - leftbound + 1  # this is the total size
        dnnz[i-ranges1[0]] = py_clip(rightbound, ranges2[0], ranges2[1]-1) - py_clip(leftbound, ranges2[0], ranges2[1]-1) + 1
    return dnnz, nnz


# This computes the preallocation between component ci1, block bi of space1 and component ci2, block bi of space2
# Where space1 and space2 are FunctionSpace (or the enriched versions)
# and interior_x, etc. indicate that an interior facet integral is happening (needed for facet integral preallocation)
# This is perfect for non-facet integrals in all dimensions, and somewhat overestimates facet integrals for 2D or 3D

# THIS IS BROKEN FOR VECTOR SPACES
# NEED TO MULTIPLY BY NDOFS (maybe ndofs^ndim?) I THINK...
# ALSO SIZING ON DNNZ AND NNZ ARRAYS IS BROKEN FOR VECTOR SPACES
def two_form_preallocate_opt(mesh, space1, space2, ci1, ci2, interior_x, interior_y, interior_z):

    elem1 = space1.themis_element()
    elem2 = space2.themis_element()

    space1_ranges = space1.get_xy(ci1)
    space2_ranges = np.array(space2.get_xy(ci2), dtype=np.int32)
    nx1s = space1.get_local_nxny(ci1)

    # swidth = space2.get_da(ci2, bi).getStencilWidth()

    # xyzmaxs_space2 = space2.get_nxny(ci2, bi)

    # compute i bounds
    dnnz_x, nnz_x = compute_1d_bounds(ci1, ci2, 0, elem1, elem2, mesh.nxs[0], nx1s[0], interior_x, mesh.bcs[0], space1_ranges[0], space2_ranges[0])

    # compute j bounds
    if mesh.ndim >= 2:
        dnnz_y, nnz_y = compute_1d_bounds(ci1, ci2, 1, elem1, elem2, mesh.nxs[1], nx1s[1], interior_y, mesh.bcs[1], space1_ranges[1], space2_ranges[1])

    # compute k bounds
    if mesh.ndim == 3:
        dnnz_z, nnz_z = compute_1d_bounds(ci1, ci2, 2, elem1, elem2, mesh.nxs[2], nx1s[2], interior_z, mesh.bcs[2], space1_ranges[2], space2_ranges[2])

    # fix periodic boundaries when there is only 1 processor in that direction
    nprocs = mesh._cell_da.getProcSizes()
    # for bc,nproc in zip(mesh.bcs,nprocs):
    if (mesh.bcs[0] == 'periodic') and (nprocs[0] == 1):
        dnnz_x = nnz_x
    if mesh.ndim >= 2:
        if (mesh.bcs[1] == 'periodic') and (nprocs[1] == 1):
            dnnz_y = nnz_y
    if mesh.ndim == 3:
        if (mesh.bcs[2] == 'periodic') and (nprocs[2] == 1):
            dnnz_z = nnz_z

    if mesh.ndim == 1:
        dnnzarr = dnnz_x
        nnzarr = nnz_x
    if mesh.ndim == 2:
        # dnnzarr = reduce(np.multiply, np.ix_(dnnz_y,dnnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
        # nnzarr = reduce(np.multiply, np.ix_(nnz_y,nnz_x)) #np.ravel(np.outer(nnz_x,nnz_y))
        dnnzarr = functools.reduce(np.multiply, np.ix_(dnnz_y, dnnz_x))  # np.ravel(np.outer(nnz_x,nnz_y))
        nnzarr = functools.reduce(np.multiply, np.ix_(nnz_y, nnz_x))  # np.ravel(np.outer(nnz_x,nnz_y))
    if mesh.ndim == 3:
        # dnnzarr = reduce(np.multiply, np.ix_(dnnz_z,dnnz_y,dnnz_x))
        # nnzarr = reduce(np.multiply, np.ix_(nnz_z,nnz_y,nnz_x))
        dnnzarr = functools.reduce(np.multiply, np.ix_(dnnz_z, dnnz_y, dnnz_x))
        nnzarr = functools.reduce(np.multiply, np.ix_(nnz_z, nnz_y, nnz_x))
    onnzarr = nnzarr - dnnzarr

    # see http://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors

    #print(dnnzarr)
    #print(onnzarr)

    return dnnzarr, onnzarr


# this is a helper function to create a TwoForm, given a UFL Form
# Mostly intended for applications that want to use Themis/UFL to handle the creation and assembly of a PETSc matrix, and then do something with it
# Doesn't interact with solver stuff, although this might be subject to change at some point

def assemble(f, bcs=None, form_compiler_parameters=None, mat_type='aij'):
    import ufl
    from solver import _extract_bcs
    from form import TwoForm

    if not isinstance(f, ufl.Form):
        raise TypeError("Provided 2-Form is a '%s', not a Form" % type(f).__name__)

    if len(f.arguments()) != 2:
        raise ValueError("Provided 2-Form is not a bilinear form")

    bcs = _extract_bcs(bcs)

# FIX HOW FORM COMPILER PARAMETERS ARE HANDLED

    form = TwoForm(f, None, mat_type=mat_type, bcs=bcs, constantJ=True, constantP=True)
    form._assemblehelper(form.mat, form.mat_type, form.mat_local_assembly_kernels.keys(), form.mat_local_assembly_kernels.values())
    form.Jassembled = True

    return form
