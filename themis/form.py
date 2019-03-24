
from petscshim import PETSc
import numpy as np
from assembly import AssembleTwoForm, AssembleOneForm, AssembleZeroForm
from tsfc_interface import compile_form
from function import Function
from ufl_expr import adjoint
import ufl
from pyop2.sparsity import get_preallocation
import time

# 3


def create_matrix(form, mat_type, target, source, blocklist, kernellist):
    # block matrix
    if mat_type == 'nest' and (target.nspaces > 1 or source.nspaces > 1):
        # create matrix array
        matrices = []
        for si1 in range(target.nspaces):
            matrices.append([])
            for si2 in range(source.nspaces):
                if ((si1, si2) in blocklist):
                    bindex = blocklist.index((si1, si2))
                    mat = create_mono(target.get_space(si1), source.get_space(si2), [(0, 0), ], [kernellist[bindex], ])
                else:
                    mat = create_empty(target.get_space(si1), source.get_space(si2))
                matrices[si1].append(mat)

        # do an empty assembly
        for si1 in range(target.nspaces):
            for si2 in range(source.nspaces):
                if ((si1, si2) in blocklist):
                    bindex = blocklist.index((si1, si2))
                    fill_mono(matrices[si1][si2], target.get_space(si1), source.get_space(si2), [(0, 0), ], [kernellist[bindex], ], zeroassembly=True)
                    # this catches bugs in pre-allocation and the initial assembly by locking the non-zero structure
                    matrices[si1][si2].setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
                    matrices[si1][si2].setOption(PETSc.Mat.Option.UNUSED_NONZERO_LOCATION_ERR, False)
                    matrices[si1][si2].setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

                    # These are for zeroRows- the first keeps the non-zero structure when zeroing rows, the 2nd tells PETSc that the process only zeros owned rows
                    matrices[si1][si2].setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
                    matrices[si1][si2].setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, False)

        # create nest
        # PETSc.Sys.Print(matrices)
        mat = PETSc.Mat().createNest(matrices, comm=PETSc.COMM_WORLD)

    # monolithic matrix
    if (mat_type == 'nest' and (target.nspaces == 1 and source.nspaces == 1)) or mat_type == 'aij':
        # create matrix
        mat = create_mono(target, source, blocklist, kernellist)
        # do an empty assembly
        fill_mono(mat, target, source, blocklist, kernellist, zeroassembly=True)

    mat.assemble()

    # this catches bugs in pre-allocation and the initial assembly by locking the non-zero structure
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
    mat.setOption(PETSc.Mat.Option.UNUSED_NONZERO_LOCATION_ERR, False)
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    # These are for zeroRows- the first keeps the non-zero structure when zeroing rows, the 2nd tells PETSc that the process only zeros owned rows
    mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
    mat.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, False)

    # print('zeroed')
    # mat.view()
    return mat


def create_empty(target, source):
    # create matrix
    mlist = []
    nlist = []
    for si1 in range(target.nspaces):
        tspace = target.get_space(si1)
        for ci1 in range(tspace.ncomp):
            m = tspace.get_localndofs(ci1)
            mlist.append(m)
    for si2 in range(source.nspaces):
        sspace = source.get_space(si2)
        for ci2 in range(sspace.ncomp):
            n = sspace.get_localndofs(ci2)
            nlist.append(n)

    M = np.sum(np.array(mlist, dtype=np.int32))
    N = np.sum(np.array(nlist, dtype=np.int32))
    mat = PETSc.Mat()
    mat.create(PETSc.COMM_WORLD)
    mat.setSizes(((M, None), (N, None)))
    mat.setType('aij')
    mat.setLGMap(target.get_overall_lgmap(), cmap=source.get_overall_lgmap())
    mat.setUp()
    mat.assemblyBegin()
    mat.assemblyEnd()

    return mat


def create_mono(target, source, blocklist, kernellist):
    # create matrix
    mlist = []
    nlist = []
    for si1 in range(target.nspaces):
        tspace = target.get_space(si1)
        for ci1 in range(tspace.ncomp):
            m = tspace.get_localndofs(ci1)
            mlist.append(m)
    for si2 in range(source.nspaces):
        sspace = source.get_space(si2)
        for ci2 in range(sspace.ncomp):
            n = sspace.get_localndofs(ci2)
            nlist.append(n)

    M = np.sum(np.array(mlist, dtype=np.int32))
    N = np.sum(np.array(nlist, dtype=np.int32))

    # Use Preallocation matrix class
    preallocator = PETSc.Mat()
    preallocator.create(PETSc.COMM_WORLD)
    preallocator.setSizes(((M, None), (N, None)))
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    preallocator.setLGMap(target.get_overall_lgmap(), cmap=source.get_overall_lgmap())
    preallocator.setUp()
    fill_mono(preallocator, target, source, blocklist, kernellist, zeroassembly=True)
    preallocator.assemble()
    dnnzarr, onnzarr = get_preallocation(preallocator, M)

    # preallocate matrix
    mat = PETSc.Mat()
    mat.create(PETSc.COMM_WORLD)
    mat.setSizes(((M, None), (N, None)))
    mat.setType('aij')
    mat.setLGMap(target.get_overall_lgmap(), cmap=source.get_overall_lgmap())
    mat.setPreallocationNNZ((dnnzarr, onnzarr))
    mat.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    mat.setUp()
    mat.zeroEntries()

    preallocator.destroy()
    return mat


def get_interior_flags(mesh, kernellist):
    interior_x = False
    interior_y = False
    interior_z = False
    for kernel in kernellist:
        if kernel.integral_type == 'interior_facet':
            interior_x = True
            interior_y = True
            interior_z = True
        if kernel.integral_type == 'interior_facet_horiz':
            interior_x = True
            if mesh.extrusion_dim == 2:
                interior_y = True
            interior_z = False
        if kernel.integral_type == 'interior_facet_vert':
            if mesh.extrusion_dim == 1:
                interior_y = True
            if mesh.extrusion_dim == 2:
                interior_z = True
    return interior_x, interior_y, interior_z


def fill_mono(mat, target, source, blocklist, kernellist, zeroassembly=False):
    # print blocklist
    for si1 in range(target.nspaces):
        isrow = target.get_field_lis(si1)
        for si2 in range(source.nspaces):
            iscol = source.get_field_lis(si2)
            if (si1, si2) in blocklist:
                submat = get_block(mat, isrow, iscol)
                bindex = blocklist.index((si1, si2))
                for kernel in kernellist[bindex]:
                    AssembleTwoForm(submat, target.get_space(si1), source.get_space(si2), kernel, zeroassembly=zeroassembly)
                restore_block(isrow, iscol, mat, submat)

    mat.assemble()


def get_block(matrix, isrow, iscol):
    if (isrow is None) or (iscol is None):
        return matrix
    else:
        return matrix.getLocalSubMatrix(isrow, iscol)


def restore_block(isrow, iscol, matrix, submat):
    if (isrow is None) or (iscol is None):
        pass
    else:
        matrix.restoreLocalSubMatrix(isrow, iscol, submat)

########################


class OneForm():

    def __init__(self, F, activefield, bcs=[], pre_f_callback=None):
        self.F = F
        self.bcs = bcs
        self._pre_f_callback = pre_f_callback

        self.space = self.F.arguments()[0].function_space()
        # NAMES?

        self.activefield = activefield

        # create vector
        self.vector = self.space.get_composite_da().createGlobalVec()
        self.vector.set(0.0)
        self.lvectors = []
        for si in range(self.space.nspaces):
            for ci in range(self.space.get_space(si).ncomp):
                self.lvectors.append(self.space.get_space(si).get_da(ci).createLocalVector())

        # compile local assembly kernels
        idx_kernels = compile_form(F)
        self.local_assembly_idx = []
        self.local_assembly_kernels = []
        for idx, kernels in idx_kernels:
            self.local_assembly_idx.append(idx)
            self.local_assembly_kernels.append(kernels)

    # BROKEN
    def output(self, view, ts=None):
        pass

    def assembleform(self, snes, X, F):

        # "copy" X (petsc Vec) into "active" field
        self.activefield._activevector = X

        # pre-function callback
        if not (self._pre_f_callback is None):
            self._pre_f_callback(X)

        # assemble
        self._assemblehelper(F)

        # restore the old active field
        self.activefield._activevector = self.activefield._vector

        # assemble X and F
        X.assemble()
        F.assemble()

    def _assemblehelper(self, vec):

        # zero out vector
        vec.set(0.0)
        for lvec in self.lvectors:
            lvec.set(0.0)

        # assemble
        blocklist = self.local_assembly_idx
        kernellist = self.local_assembly_kernels
        for si1 in range(self.space.nspaces):
            soff = self.space.get_space_offset(si1)
            if (si1,) in blocklist:
                bindex = blocklist.index((si1,))
                for kernel in kernellist[bindex]:
                    AssembleOneForm(self.lvectors[soff:soff+self.space.get_space(si1).ncomp], self.space.get_space(si1), kernel)

        self.space.get_composite_da().gather(vec, PETSc.InsertMode.ADD_VALUES, self.lvectors)

        # Apply symmetric essential boundaries
        # set residual to ZERO at boundary points, since we set x(boundary) = bvals in non-linear solve and also apply boundary conditions to Jacobian
        for bc in self.bcs:
            bc.apply_vector(vec, zero=True)

    def destroy(self):
        self.vector.destroy()
        for v in self.lvectors:
            v.destroy()


class Matrix():

    def __init__(self, form, target, source, bcs, mat_type):
        self._form = form
        self._bcs = bcs
        self._target = target
        self._source = source
        self._mat_type = mat_type
        self._assembled = False

        idx_kernels = compile_form(self._form)
        self.mat_local_assembly_idx = []
        self.mat_local_assembly_kernels = []
        for idx, kernels in idx_kernels:
            self.mat_local_assembly_idx.append(idx)
            self.mat_local_assembly_kernels.append(kernels)

        self.petscmat = create_matrix(self, mat_type, target, source, self.mat_local_assembly_idx, self.mat_local_assembly_kernels)

        for bc in self._bcs:
            rowlgmap, collgmap = self.petscmat.getLGMap()
            rowindices = rowlgmap.getIndices()
            colindices = collgmap.getIndices()
            bcind = bc.get_boundary_indices_local()
            rowindices[bcind] = -1
            colindices[bcind] = -1

    def assemble(self, mat):

        # zero out matrix
        mat.zeroEntries()

        # assemble
        fill_mono(mat, self._target, self._source, self.mat_local_assembly_idx, self.mat_local_assembly_kernels)

        # set boundary rows equal to zero
        for bc in self._bcs:
            bc.apply_mat(mat, self._mat_type)

        self._assembled = True

# DOES DESTROYING A NEST AUTOMATICALLY DESTROY THE SUB MATRICES?
    def destroy(self):
        self.petscmat.destroy()


class ImplicitMatrix():

    def __init__(self, form, target, source, rowbcs, colbcs):
        self._twoform = form
        self._rowbcs = rowbcs
        self._target = target
        self._source = source
        self._colbcs = colbcs

        self._x = Function(target, name='internalx')
        self._y = Function(source, name='internaly')

        self._form = ufl.action(form, self._x)
        self._formT = ufl.action(adjoint(form), self._y)
        self._assembled = False

        if len(rowbcs) > 0:
            self._xbcs = Function(target)
        if len(colbcs) > 0:
            self._ybcs = Function(source)

        idx_kernels = compile_form(self._form)
        self.local_assembly_idx = []
        self.local_assembly_kernels = []
        for idx, kernels in idx_kernels:
            self.local_assembly_idx.append(idx)
            self.local_assembly_kernels.append(kernels)

        idx_kernels = compile_form(self._formT)
        self.Tlocal_assembly_idx = []
        self.Tlocal_assembly_kernels = []
        for idx, kernels in idx_kernels:
            self.Tlocal_assembly_idx.append(idx)
            self.Tlocal_assembly_kernels.append(kernels)

        # create matrix
        mlist = []
        nlist = []
        for si1 in range(target.nspaces):
            tspace = target.get_space(si1)
            for ci1 in range(tspace.ncomp):
                m = tspace.get_localndofs(ci1)
                mlist.append(m)
        for si2 in range(source.nspaces):
            sspace = source.get_space(si2)
            for ci2 in range(sspace.ncomp):
                n = sspace.get_localndofs(ci2)
                nlist.append(n)

        M = np.sum(np.array(mlist, dtype=np.int32))
        N = np.sum(np.array(nlist, dtype=np.int32))
        self.petscmat = PETSc.Mat()
        self.petscmat.create(PETSc.COMM_WORLD)
        self.petscmat.setSizes(((M, None), (N, None)))
        self.petscmat.setType('python')
        self.petscmat.setPythonContext(self)
        self.petscmat.setUp()
        self.petscmat.assemblyBegin()
        self.petscmat.assemblyEnd()

    def mult(self, A, x, y):

        # "copy" X (petsc Vec) into "active" field
        self._x._activevector = x

        # zero out local y
        y.set(0.0)
        for lvec in self._y._lvectors:
            lvec.set(0.0)

        # save xbcs
        if len(self._rowbcs) > 0:
            x.copy(result=self._xbcs._activevector)

        # apply zero bcs to x
        # sets x = [ xint   0  ]
        for bc in self._colbcs:
            bc.apply_vector(x, zero=True)

        # compute multiplication  A * x
        # This gives y = [ Aint xint    Junk  ] since we set xbc = 0
        blocklist = self.local_assembly_idx
        kernellist = self.local_assembly_kernels
        for si1 in range(self._target.nspaces):
            soff = self._target.get_space_offset(si1)
            if (si1,) in blocklist:
                bindex = blocklist.index((si1,))
                for kernel in kernellist[bindex]:
                    AssembleOneForm(self._y._lvectors[soff:soff+self._target.get_space(si1).ncomp], self._target.get_space(si1), kernel)

        self._target.get_composite_da().gather(y, PETSc.InsertMode.ADD_VALUES, self._y._lvectors)

        # apply bcs to y
        # this corrects Junk
        for bc in self._rowbcs:
            bc.apply_vector(y, bvals=self._xbcs)

        # y.view()
        # restore active field
        self._x._activevector = self._x._vector

    def multTranspose(self, A, y, x):

        # "copy" y (petsc Vec) into "active" field
        self._y._activevector = y

        # zero out local x
        x.set(0.0)
        for lvec in self._x._lvectors:
            lvec.set(0.0)

        # save ybcs
        if len(self._colbcs) > 0:
            y.copy(result=self._ybcs._activevector)

        # apply zero bcs to y
        # sets y = [ yint   0  ]
        for bc in self._rowbcs:
            bc.apply_vector(y, zero=True)

        # compute multiplication  A * y
        # This gives x = [ Aint yint    Junk  ] since we set ybc = 0
        blocklist = self.Tlocal_assembly_idx
        kernellist = self.Tlocal_assembly_kernels
        for si1 in range(self._source.nspaces):
            soff = self._source.get_space_offset(si1)
            if (si1,) in blocklist:
                bindex = blocklist.index((si1,))
                for kernel in kernellist[bindex]:
                    AssembleOneForm(self._x._lvectors[soff:soff+self._source.get_space(si1).ncomp], self._source.get_space(si1), kernel)

        self._source.get_composite_da().gather(x, PETSc.InsertMode.ADD_VALUES, self._x._lvectors)

        # apply bcs to x
        # this corrects Junk
        for bc in self._colbcs:
            bc.apply_vector(x, bvals=self._ybcs)

        # restore active field
        self._y._activevector = self._y._vector

    def assemble(self, mat):
        self.petscmat.assemble()
        self._assembled = True

    def destroy(self):
        self.petscmat.destroy()
        self._x.destroy()
        self._y.destroy()
        self._xbcs.destroy()
        self._ybcs.destroy()

# HOW IS FIELDSPLIT HANDLED?
# I see how to create implicit blocks, but what about explicit blocks
# ie it would be great to be able to apply sub matrices in a matrix-free manner and also assemble some (possibly further approximated) preconditioner?
    def createSubMatrix(self,):
        pass


class TwoForm():
    def __init__(self, J, activefield, Jp=None, mat_type='aij', pmat_type='aij', constantJ=False, constantP=False, bcs=[], pre_j_callback=None):

        self.target = J.arguments()[0].function_space()
        self.source = J.arguments()[1].function_space()
        if not (Jp is None):
            self.ptarget = Jp.arguments()[0].function_space()
            self.psource = Jp.arguments()[1].function_space()
        # ADD A CHECK HERE THAT PSOOURCE AND PTARGET CAN DIFFER ONLY FOR MGD SPACES

        self.activefield = activefield
        self._pre_j_callback = pre_j_callback

        self.bcs = bcs
        self.constantJ = constantJ
        self.constantP = constantP

        self.mat_type = mat_type
        self.pmat_type = pmat_type
        self.J = J
        self.Jp = Jp

        # create matrices
        if mat_type in ['aij', 'nest']:
            self.mat = Matrix(self.J, self.target, self.source, bcs, mat_type)
        elif mat_type == 'matfree':
            self.mat = ImplicitMatrix(self.J, self.target, self.source, bcs, bcs)

        # bcs are okay here because MGD degrees of freedom are the same for every order
# POSSIBLY RECONSTRUCT BCS HERE?
# IE I COULD SEE THIS BECOMING A PROBLEM?
# This should maybe actually be a python preconditioner?
        if not (self.Jp is None):
            if pmat_type in ['aij', 'nest']:
                self.pmat = Matrix(self.Jp, self.ptarget, self.psource, bcs, pmat_type)
            elif pmat_type == 'matfree':
                raise ValueError('dont know how to treat an ImplicitMatrix pmat yet!')
# ISSUE HERE WITH DIFFERENT SPACES FOR J and Jp since active field is not valid anymore!
                # self.pmat = ImplicitMatrix(self.Jp, self.ptarget, self.psource, bcs, bcs)

    def destroy(self):
        self.mat.destroy()
        if not (self.Jp is None):
            self.pmat.destroy()

    def assembleform(self, snes, X, J, P):

        # "copy" X into "active" field
        self.activefield._activevector = X

        if (not (self.constantJ and self.mat._assembled)) or ((self.Jp is not None) and (not (self.constantP and self.pmat._assembled))):
            # pre-jacobian callback
            if self._pre_j_callback is not None:
                self._pre_j_callback(X)

        # assemble J unless J is constant and assembled
        if not (self.constantJ and self.mat._assembled):
            self.mat.assemble(J)

        # assemble Jp IF it exists unless Jp is constant and assembled
        if (self.Jp is not None) and (not (self.constantP and self.pmat._assembled)):
            self.pmat.assemble(P)

        # restore the old active field
        self.activefield._activevector = self.activefield._vector

        # PETSc.Sys.Print('mat', self.mat.petscmat.getInfo(info=3))
        # if not (self.Jp is None):
        #   PETSc.Sys.Print('pmat', self.pmat.petscmat.getInfo(info=3))

        # WHAT SHOULD I REALLY BE RETURNING HERE?
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class ZeroForm():

    def __init__(self, E):
        self.E = E
        idx_kernels = compile_form(E)
        self.kernellist = idx_kernels[0][1]  # This works because we only have 1 idx (0-forms have no arguments) and therefore only 1 kernel list
        self.value = 0.
        self.mesh = self.E.ufl_domain()

    def assembleform(self):
        return AssembleZeroForm(self.mesh, self.kernellist)
