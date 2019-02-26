import numpy as np
from form import get_block, restore_block
from petscshim import PETSc
from function import Function
from constant import Constant
import ufl
from project import Projector


class DirichletBC():

    def __init__(self, space, val, subdomain, method='topological'):
        if not method == 'topological':
            raise ValueError('%s method for setting boundary nodes is not supported', method)

        if isinstance(val, Function) and val.function_space() != space:
            raise RuntimeError("%r is defined on incompatible FunctionSpace!" % val)

        self._space = space
        self._val = val

        # turn subdomains into a set of boundary nodes and create bvals and zerovals arrays

        self._splitfieldglobalrows = []
        self._globalrows = []
        self._localrows = []
        self._bvals = []
        self._zerovals = []

        direclist = space._mesh.get_boundary_direcs(subdomain)
        self.si = space._si

        localrowoffset = 0
        # globalrowoffset = 0
        for i in range(0, self.si):
            for ci in range(space._parent.get_space(i).ncomp):
                localrowoffset = localrowoffset + space._parent.get_space(i).get_localghostedndofs(ci)
                # globalrowoffset = globalrowoffset + space._parent.get_space(i).get_localndofs(ci)
                # PETSc.Sys.Print(space._parent.get_space(self.si).get_localndofs(ci, 0),space._parent.get_space(self.si).get_localghostedndofs(ci, 0))
                # localrowoffset = localrowoffset + space._parent.get_space(self.si).get_localndofs(ci, 0)

        # PETSc.Sys.Print(space,space._si,localrowoffset,globalrowoffset)

        for direc in direclist:
            for ci in range(space.get_space(self.si).ncomp):

                rows = space.get_space(self.si).get_boundary_indices(ci, direc)  # returns split component local rows
                if rows[0] == -1:
                    rows = np.array([], dtype=np.int32)

                # PETSc.Sys.Print(direc,ci,rows,space.get_component_compositelgmap(self.si, ci, bi).apply(rows))

                split_global_rows = space._parent.get_space(self.si).get_component_compositelgmap(self.si, ci).apply(rows)
                global_rows = space._parent.get_component_compositelgmap(self.si, ci).apply(rows)
                self._splitfieldglobalrows.append(split_global_rows)  # turns rows from split component local into split field global
                self._globalrows.append(global_rows)  # turn rows from split component local into monolithic global
                self._localrows.append(rows + localrowoffset)

                if isinstance(val, (int, float)):
                    valarr = np.ones(len(rows)) * val
                elif isinstance(val, Constant):
                    valarr = np.ones(len(rows)) * float(val)
                elif isinstance(val, Function):
                    valarr = val._vector.getValues(global_rows)
                elif isinstance(val, ufl.core.expr.Expr):
                    valfunc = Function(space)
                    projector = Projector(val, valfunc)
                    projector.project()
                    valarr = valfunc._vector.getValues(global_rows)
                else:
                    raise ValueError('dont know how to handle val of type', type(val))

                if len(rows) == 0:
                    self._bvals.append([])
                    self._zerovals.append([])
                else:
                    self._bvals.append(valarr)
                    self._zerovals.append(np.zeros(len(rows)))

        if not (len(direclist) == 0):
            self._splitfieldglobalrows = np.concatenate(self._splitfieldglobalrows)
            self._globalrows = np.concatenate(self._globalrows)
            self._localrows = np.concatenate(self._localrows)
            self._zerovals = np.concatenate(self._zerovals)
            self._bvals = np.concatenate(self._bvals)

        # PETSc.Sys.Print(self._localrows)
        # PETSc.Sys.Print(self._globalrows)

    def get_boundary_indices_local(self):
        return self._localrows

    def get_boundary_indices_global(self):
        return self._globalrows

    # IS THIS NEEDED?
    # def get_boundary_indices_splitglobal(self):
    #     return self._splitfieldglobalrows

    def apply_vector(self, vector, zero=False):
        if zero:
            vector.setValues(self._globalrows, self._zerovals, addv=PETSc.InsertMode.INSERT_VALUES)
        else:
            vector.setValues(self._globalrows, self._bvals, addv=PETSc.InsertMode.INSERT_VALUES)
        vector.assemble()

    def apply_mat(self, mat, mat_type):

        if mat_type == 'aij':
            mat.zeroRows(self._globalrows, 1.0)

        if mat_type == 'nest':
            space = self._space._parent
            isrow = space.get_field_lis(self.si)
            for si in range(space.nspaces):
                iscol = space.get_field_lis(si)
                submat = get_block(mat, isrow, iscol)
                if (si == self.si):
                    submat.zeroRows(self._splitfieldglobalrows, 1.0)
                else:
                    submat.zeroRows(self._splitfieldglobalrows, 0.0)
                restore_block(isrow, iscol, mat, submat)

# class DirichletBC():

    # # THIS DOES NOT SUPPORT SETTING BCS ON COMPONENTS OF A VECTOR SPACE

    # # THIS IS CURRENTLY ONLY HOMOGENOUS BCS (AND ONLY CORRECT FOR NON-ZERO WHEN BASIS IS NODAL)
    # # EVENTUALLY THIS SHOULD TAKE FOR VALS EITHER
    # # 1) A LITERAL CONSTANT OF THE APPROPRIATE RANK (WHICH MIGHT BE PROJECTED INTO SPACE IFF THE SPACE IS NOT NODAL)
    # # 2) A FUNCTION (IN SPACE)
    # # 3) A UFL EXPRESSION (THAT IS THEN PROJECTED INTO SPACE)
    # # 4) A CONSTANT (WHICH MIGHT HAVE TO BE PROJECTED INTO SPACE IFF THE SPACE IS NOT NODAL)

    # def __init__(self, space, si, val, direc):

        # # ADD CHECKS ON VAL
        # # ADD CHECKS ON DIREC

        # self.si = si

        # self._direc = direc
        # self._mesh = space.mesh()
        # self._space = space

        # # ADD CHECKS THAT DIREC DOESN'T CONFLICY WITH MESH BCS IE TRYING TO SET A BC ON A PERIODIC BOUNDARY...

        # bi = 0  # FIX FOR MULTIPATCH...

        # # MIGHT NOT HAVE A NODAL BASIS, IN WHICH CASE WE NEED TO PROJECT THE VALUE WE RECEIVE!
        # # VAL HANDLING IS BROKEN/IGNORED RIGHT NOW ANYWAYS

        # # CAN SUPPORT CLEVERER TYPES OF DIREC BY JUST SUMMING OVER VARIOUS THINGS HERE...
        # # EXAMPLE: DIREC=ALL FOR 2D GIVES X+,X-,Y+,Y-

        # self._splitfieldglobalrows = []
        # self._globalrows = []
        # self._bvals = []
        # self._zerovals = []
        # for ci in range(space.get_space(self.si).ncomp):

            # rows = space.get_space(self.si).get_boundary_indices(ci, bi, direc)  # returns split component local rows
            # if rows[0] == -1:
                # rows = []

            # self._splitfieldglobalrows.append(space.get_space(self.si).get_component_compositelgmap(self.si, ci, bi).apply(rows))  # turns rows from split component local into split field global
            # self._globalrows.append(space.get_component_compositelgmap(self.si, ci, bi).apply(rows))  # turn rows from split component local into monolithic global

            # if len(rows) == 0:
                # self._bvals.append([])
                # self._zerovals.append([])
            # else:
                # self._bvals.append(np.ones(len(rows)) * 0.)  # WE CURRENTLY IGNORE VAL...
                # self._zerovals.append(np.zeros(len(rows)))

        # self._splitfieldglobalrows = np.concatenate(self._splitfieldglobalrows)
        # self._globalrows = np.concatenate(self._globalrows)
        # self._zerovals = np.concatenate(self._zerovals)
        # self._bvals = np.concatenate(self._bvals)

    # def space(self):
        # return self._space

    # # THIS CAN BE OPTIMIZED BY INSTEAD MODIFYING INDIRECTION MAPS TO EQUAL -1 (WHICH TELLS PETSC TO DROP THE VALUE)
    # # THEN IN FORM.PY WE WOULD CALL APPLY MATRIX BEFORE ASSEMBLING
    # # THIS AVOIDS EXPENSIVE SEARCHES THROUGH COLUMNS TO ZERO STUFF
    # def apply_matrix(self, mat, mattype):

        # # DOES THIS ACTUALLY WORK FOR NEST MATRICES- NOT REALLY SINCE ZEROING COLUMNS REQUIRES SEARCHING THROUGH THE WHOLE MATRIX, NOT JUST PART OF IT!
        # if mattype == 'nest':
            # isrow = self._space.get_field_lis(self.si)
            # for si in range(self._space.nspaces):
                # # sspace = self._space.get_space(si)
                # iscol = self._space.get_field_lis(si)
                # submat = get_block(mat, isrow, iscol)
                # if (si == self.si):
                # submat.zeroRowsColumns(self._splitfieldglobalrows, 1.0)
                # # submat.zeroRows(self._splitfieldglobalrows,1.0)
                # else:
                # submat.zeroRowsColumns(self._splitfieldglobalrows, 0.0)
                # # submat.zeroRows(self._splitfieldglobalrows,0.0)
                # restore_block(isrow, iscol, mat, submat)

        # if mattype == 'aij':  # or (form.mattype == 'block' and (len(form.matrices) == 1) and (len(form.matrices[0]) == 1)):
            # mat.zeroRowsColumns(self._globalrows, 1.0)  # + self.space.get_space(self.si).localoffsets[self.ci]
            # # mat.zeroRows(self._globalrows,1.0) #  + self.space.get_space(self.si).localoffsets[self.ci]

        # # HOW DOES THIS WORK FOR MATRIX-FREE?

    # def apply_vector(self, vector, zero=False):
        # if zero:
            # vector.setValues(self._globalrows, self._zerovals, addv=PETSc.InsertMode.INSERT_VALUES)
        # else:
            # vector.setValues(self._globalrows, self._bvals, addv=PETSc.InsertMode.INSERT_VALUES)
        # vector.assemble()
