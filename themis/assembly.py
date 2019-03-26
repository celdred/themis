
import numpy as np
from themis.petscshim import PETSc
from mpi4py import MPI
import time
from themis.assembly_utils import extract_fields, compile_functional

__all__ = ["assemble", "AssembleOneForm", "AssembleTwoForm", "AssembleZeroForm"]

def AssembleTwoForm(mat, tspace, sspace, kernel, zeroassembly=False):

    # FIX THIS- SHOULD BE TIED TO KERNEL!
    name = 'test'

    with PETSc.Log.Stage(name + '_assemble'):
        mesh = tspace.mesh()

        if zeroassembly:
            kernel.zero = True
        # compile functional IFF form not already compiled
        # also extract coefficient and geometry args lists
        # time1 = time.time()
        with PETSc.Log.Event('compile'):
            if not kernel.assemblycompiled:
                compile_functional(kernel, tspace, sspace, mesh)
        # print('compiled-2',time.time()-time1,zeroassembly)

        # scatter fields into local vecs
        # time1 = time.time()
        with PETSc.Log.Event('extract'):
            extract_fields(kernel)
        # print('extracted-2',time.time()-time1,zeroassembly)

        # assemble
        # time1 = time.time()
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
                # PETSc.Sys.Print('assembling 2-form',kernel.integral_type,submatlist)
                # PETSc.Sys.Print(submatlist)
                # PETSc.Sys.Print(tdalist)
                # PETSc.Sys.Print(sdalist)
                # PETSc.Sys.Print(kernel.fieldargs_list)
                # PETSc.Sys.Print(kernel.constantargs_list)
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
        # print('assembled-2',time.time()-time1,zeroassembly)


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
        # time1 = time.time()
        with PETSc.Log.Event('compile'):
            if not kernel.assemblycompiled:
                compile_functional(kernel, space, None, mesh)
        # print('compiled-1',time.time()-time1)

        # scatter fields into local vecs
        # time1 = time.time()
        with PETSc.Log.Event('extract'):
            extract_fields(kernel)
        # print('extracted-1',time.time()-time1)

        # assemble
        # time1 = time.time()
        with PETSc.Log.Event('assemble'):

            # get the list of das
            tdalist = []
            for ci1 in range(space.ncomp):
                tdalist.append(space.get_da(ci1))

            for da, assemblefunc in zip(kernel.dalist, kernel.assemblyfunc_list):
                # PETSc.Sys.Print('assembling 1-form',kernel.integral_type,veclist)
                    #PETSc.Sys.Print(kernel.assembly_routine)
                    # PETSc.Sys.Print(kernel.exterior_facet_type,kernel.exterior_facet_direction)
                # PETSc.Sys.Print(veclist)
                # PETSc.Sys.Print(tdalist)
                # PETSc.Sys.Print(kernel.fieldargs_list)
                # PETSc.Sys.Print(kernel.constantargs_list)
                # PETSc.Sys.Print(kernel.fieldargs_list[0].getGhostRanges())
                assemblefunc([da, ] + veclist + tdalist + kernel.fieldargs_list, kernel.constantargs_list)
        # print('assembled-1',time.time()-time1)


# this is a helper function to create a TwoForm, given a UFL Form
# Mostly intended for applications that want to use Themis/UFL to handle the creation and assembly of a PETSc matrix, and then do something with it
# Doesn't interact with solver stuff, although this might be subject to change at some point

# EVENTUALLY ALSO ADD 1-FORM ASSEMBLY HERE
# THIS WILL BE VERY USEFUL FOR MASS-LUMPED VARIANTS IE FULLY EXPLICIT TIME STEPPING WITHOUT LINEAR SOLVES...

def assemble(f, bcs=None, form_compiler_parameters=None, mat_type='aij'):
    import ufl
    from solver import _extract_bcs
    from form import TwoForm

    if not isinstance(f, ufl.Form):
        raise TypeError("Provided f is a '%s', not a Form" % type(f).__name__)

    if len(f.arguments()) != 2:
        raise ValueError("Provided f is not a bilinear form")

    bcs = _extract_bcs(bcs)

# FIX HOW FORM COMPILER PARAMETERS ARE HANDLED

    form = TwoForm(f, None, mat_type=mat_type, bcs=bcs, constantJ=True, constantP=True)
    form._assemblehelper(form.mat, form.mat_type, form.mat_local_assembly_kernels.keys(), form.mat_local_assembly_kernels.values())
    form.Jassembled = True

    return form
