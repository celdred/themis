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
# HOW DO WE HANDLE THE CASE OF SETTING COMPONENTS OF A VECTOR OR TENSOR SPACE EQUAL TO SOMETHING?
# DOES THIS EVEN REALLY MAKE SENSE?
# OR A NORMAL COMPONENT, ETC?

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
                    if space.interpolatory:
                        valfunc.interpolate(val, valfunc)
                    else:
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

    def apply_vector(self, vector, bvals=None, zero=False):
        if zero:
            vector.setValues(self._globalrows, self._zerovals, addv=PETSc.InsertMode.INSERT_VALUES)
        else:
            if bvals is None:
                vector.setValues(self._globalrows, self._bvals, addv=PETSc.InsertMode.INSERT_VALUES)
            else:
                bvalsarr = bvals._activevector.getValues(self._globalrows)
                vector.setValues(self._globalrows, bvalsarr, addv=PETSc.InsertMode.INSERT_VALUES)
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
