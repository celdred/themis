import numpy as np
from form import get_block, restore_block
from petscshim import PETSc


class DirichletBC():

    # THIS DOES NOT SUPPORT SETTING BCS ON COMPONENTS OF A VECTOR SPACE

    # THIS IS CURRENTLY ONLY HOMOGENOUS BCS (AND ONLY CORRECT FOR NON-ZERO WHEN BASIS IS NODAL)
    # EVENTUALLY THIS SHOULD TAKE FOR VALS EITHER
    # 1) A LITERAL CONSTANT OF THE APPROPRIATE RANK (WHICH MIGHT BE PROJECTED INTO SPACE IFF THE SPACE IS NOT NODAL)
    # 2) A FUNCTION (IN SPACE)
    # 3) A UFL EXPRESSION (THAT IS THEN PROJECTED INTO SPACE)
    # 4) A CONSTANT (WHICH MIGHT HAVE TO BE PROJECTED INTO SPACE IFF THE SPACE IS NOT NODAL)

    def __init__(self, space, si, val, direc):

        # ADD CHECKS ON VAL
        # ADD CHECKS ON DIREC

        self.si = si

        self._direc = direc
        self._mesh = space.mesh()
        self._space = space

        # ADD CHECKS THAT DIREC DOESN'T CONFLICY WITH MESH BCS IE TRYING TO SET A BC ON A PERIODIC BOUNDARY...

        bi = 0  # FIX FOR MULTIPATCH...

        # MIGHT NOT HAVE A NODAL BASIS, IN WHICH CASE WE NEED TO PROJECT THE VALUE WE RECEIVE!
        # VAL HANDLING IS BROKEN/IGNORED RIGHT NOW ANYWAYS

        # CAN SUPPORT CLEVERER TYPES OF DIREC BY JUST SUMMING OVER VARIOUS THINGS HERE...
        # EXAMPLE: DIREC=ALL FOR 2D GIVES X+,X-,Y+,Y-

        self._splitfieldglobalrows = []
        self._globalrows = []
        self._bvals = []
        self._zerovals = []
        for ci in range(space.get_space(self.si).ncomp):

            rows = space.get_space(self.si).get_boundary_indices(ci, bi, direc)  # returns split component local rows
            if rows[0] == -1:
                rows = []

            self._splitfieldglobalrows.append(space.get_space(self.si).get_component_compositelgmap(self.si, ci, bi).apply(rows))  # turns rows from split component local into split field global
            self._globalrows.append(space.get_component_compositelgmap(self.si, ci, bi).apply(rows))  # turn rows from split component local into monolithic global

            if len(rows) == 0:
                self._bvals.append([])
                self._zerovals.append([])
            else:
                self._bvals.append(np.ones(len(rows)) * 0.)  # WE CURRENTLY IGNORE VAL...
                self._zerovals.append(np.zeros(len(rows)))

        self._splitfieldglobalrows = np.concatenate(self._splitfieldglobalrows)
        self._globalrows = np.concatenate(self._globalrows)
        self._zerovals = np.concatenate(self._zerovals)
        self._bvals = np.concatenate(self._bvals)

    def space(self):
        return self._space

    # THIS CAN BE OPTIMIZED BY INSTEAD MODIFYING INDIRECTION MAPS TO EQUAL -1 (WHICH TELLS PETSC TO DROP THE VALUE)
    # THEN IN FORM.PY WE WOULD CALL APPLY MATRIX BEFORE ASSEMBLING
    # THIS AVOIDS EXPENSIVE SEARCHES THROUGH COLUMNS TO ZERO STUFF
    def apply_matrix(self, mat, mattype):

        # DOES THIS ACTUALLY WORK FOR NEST MATRICES- NOT REALLY SINCE ZEROING COLUMNS REQUIRES SEARCHING THROUGH THE WHOLE MATRIX, NOT JUST PART OF IT!
        if mattype == 'nest':
            isrow = self._space.get_field_lis(self.si)
            for si in range(self._space.nspaces):
                # sspace = self._space.get_space(si)
                iscol = self._space.get_field_lis(si)
                submat = get_block(mat, isrow, iscol)
                if (si == self.si):
                    submat.zeroRowsColumns(self._splitfieldglobalrows, 1.0)
                    # submat.zeroRows(self._splitfieldglobalrows,1.0)
                else:
                    submat.zeroRowsColumns(self._splitfieldglobalrows, 0.0)
                    # submat.zeroRows(self._splitfieldglobalrows,0.0)
                restore_block(isrow, iscol, mat, submat)

        if mattype == 'aij':  # or (form.mattype == 'block' and (len(form.matrices) == 1) and (len(form.matrices[0]) == 1)):
            mat.zeroRowsColumns(self._globalrows, 1.0)  # + self.space.get_space(self.si).localoffsets[self.ci]
            # mat.zeroRows(self._globalrows,1.0) #  + self.space.get_space(self.si).localoffsets[self.ci]

        # HOW DOES THIS WORK FOR MATRIX-FREE?

    def apply_vector(self, vector, zero=False):
        if zero:
            vector.setValues(self._globalrows, self._zerovals, addv=PETSc.InsertMode.INSERT_VALUES)
        else:
            vector.setValues(self._globalrows, self._bvals, addv=PETSc.InsertMode.INSERT_VALUES)
        vector.assemble()
