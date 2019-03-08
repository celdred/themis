from ufl import FunctionSpace as UFLFunctionSpace
from ufl import MixedFunctionSpace as UFLMixedFunctionSpace
from petscshim import PETSc
from themiselement import ThemisElement
import numpy as np
import ufl


class FunctionSpace(UFLFunctionSpace):

# ADD THIS
    def get_local_ndofs(self):
        return FIXME
        
    def destroy(self):
        self._da.destroy()

    def output(self):
        pass

    # DAs
    # return the da
    def get_da(self):
        return self._da

    # LGMAPS
    # returns the lgmap that goes from (split) local indices to (split) global indices
    def get_lgmap(self,):
        return self._lgmap
    
    def get_field_lgmap(self,si):
        return self._lgmap
        
# THIS NOW BROKEN
    # DOES THIS WORK FOR VECTOR SPACES?
    # NO, IT DOESN'T TAKE INTO ACCOUNT THAT NDOF IS NOT EQUAL TO 1...
    # returns the SPLIT LOCAL indices of the owned portion of the physical boundary for component ci in direc
    def get_boundary_indices(self, direc):

# Do this in multiple steps
# 1) get boundary elements for a given boundary ie direc that this process owns (THIS IS DONE BY THE MESH!)
# 2) compute size of boundary indices array (based on the number of boundary elements)
# 3) loop over boundary elements and add local indices (do this in python to start, should be ok for 1D/2D, but probably want to do it in cython/C for 3D...)
        
        #ALSO HANDLE VECTOR SPACES!
        
        #MAYBE THESE SHOULD BE PRE-COMPUTED WHEN SPACES ARE CREATED?
        #YES
        xs,xm = self.mesh.boundary_element_indices(direc)
        if not xs is None:
            boundary_size = np.prod(xm,dtype=np.int32)
            ndim = self.mesh.ndim
            dof0,dof1,dof2,dof3 = self._themiselement.dofmap()
            if ndim == 1: boundary_dofs = dof0
            if ndim == 2: boundary_dofs = dof0 + dof1
            if ndim == 3: boundary_dofs = dof0 + 2*dof1 + dof2
            
            # This incorporates dummy elements...
            boundary_indices = np.zeros((boundary_size,boundary_dofs),dtype=np.int32)
            for k in range(xs[2],xs[2]+xm[2]):
                for j in range(xs[1],xs[1]+xm[1]):
                    for i in range(xs[0],xs[0]+xm[0]):
                        d = 0
                        if ndim == 1:
                            for l in range(dof0):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.LEFT,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
                        if ndim == 2:
                            for l in range(dof0):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN_LEFT,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
                            for l in range(dof1):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1                                                        
                        if ndim == 3: #FIX THIS- STENCIL LOCATIONS ARE BROKEN
                            for l in range(dof0):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN_LEFT,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
                            for l in range(dof1):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
                            for l in range(dof2):
                                idx = self.mesh.da.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN,l)
                                boundary_indices[i + j*xs[0] + k*xs[0]*xs[1]][d] = SOMETHING
                                d = d + 1
            return np.ravel(boundary_indices)
        else:
            return np.array([],dtype=np.int32)
            
    #############################
    # These are included to simplify code: they allow FunctionSpace to act as if they are a MixedFunctionSpace of size 1
    # returns the space for field si of a (mixed) space
    def get_space(self, si):
        return self

    # return the local field IS for field si- these are used for 2-form assembly
    # The None flag tells getblock that we should actually return the Matrix itself, not a submatrix
    def get_field_lis(self, si):
        return None

    #######################################################

    def mesh(self):
        return self.ufl_domain()

    def __len__(self):
        return 1

    def themis_element(self):
        return self._themiselement

    def split(self):
        """Split into a tuple of constituent spaces."""
        return (self, )

# HOW SHOULD OUTPUT/NAMES BE HANDLED HERE?
    def __init__(self, mesh, element, name='fspace', si=0, parent=None):

        self._themiselement = ThemisElement(element)
        self._uflelement = element

        self._mesh = mesh
        self._name = name
        self._spaces = [self, ]
        self.nspaces = 1

        self._si = si
        if parent is None:
            self._parent = self
        else:
            self._parent = parent

        UFLFunctionSpace.__init__(self, mesh, element)

        # create das and lgmaps
        self._da = mesh.create_dof_map(self._themiselement)
        self._lgmap = self._da.getLGMap()

class MixedFunctionSpace(UFLMixedFunctionSpace):

    def mesh(self):
        return self.ufl_domain()

    def __len__(self):
        """Return the number of :class:`FunctionSpace` of which this
        :class:`MixedFunctionSpace` is composed."""
        return self.nspaces

    def sub(self, i):
        return self._spaces[i]

    def split(self):
        return self._spaces

    def __init__(self, spacelist, name='mixedspace'):
        # self._spaces = spacelist
        self.nspaces = len(spacelist)

        # ADD CHECK THAT ALL SPACES ARE DEFINED ON THE SAME MESH
        self._spaces = []
        for i, space in enumerate(spacelist):
            self._spaces.append(FunctionSpace(spacelist[i]._mesh, spacelist[i]._uflelement, name=name+str(i), si=i, parent=self))

        UFLMixedFunctionSpace.__init__(self, *spacelist)

        # create composite DM
        self._composite_da = PETSc.DMComposite().create()
        for si in range(self.nspaces):
            self._composite_da.addDM(self.get_space(si).get_da())
        self._composite_da.setUp()

        self._field_lis = self._composite_da_field.getLocalISs()
        self._field_gis = self._omposite_da_field.getGlobalISs()
        
        self._lgmap = self._composite_da.getLGMap()
        self._field_lgmaps = self._composite_da.getLGMaps()

    # return the overall lgmap for the mixed space
    def get_overall_lgmap(self):
        return self._overall_lgmap

    def get_composite_da(self):
        return self._composite_da

    # returns the space for field j of a (mixed) space
    def get_space(self, j):
        return self._spaces[j]

    # These operate on the split field-wise
    # returns the lgmap that 
    # This is used exclusively for boundary conditions to transforms boundary indices from split local to split global
    def get_field_lgmap(self, si):
        return self._field_lgmaps[si]

    # These operate on the split FIELD-WISE spaces
    # returns the global IS for space si- these map from split global to monolithic global
    # They are used for setting fieldsplits
    def get_field_gis(self, si):
        return self._field_gis[si]  # 

    # return the local IS for space si- these are used in MatGetLocalSubMatrix
    def get_field_lis(self, si):
        return self._field_lis[si]  # 


def make_scalar_element(mesh, family, degree, vfamily, vdegree):
    """Build a scalar :class:`ufl.FiniteElement`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the element is returned immediately.
    """

    if isinstance(family, ufl.FiniteElementBase):
        return family

    cell = mesh.ufl_cell()

    if isinstance(cell, ufl.TensorProductCell) \
       and vfamily is not None and vdegree is not None:
        la = ufl.FiniteElement(family,
                               cell=cell.sub_cells()[0],
                               degree=degree)
        # If second element was passed in, use it
        lb = ufl.FiniteElement(vfamily,
                               cell=ufl.interval,
                               degree=vdegree)
        # Now make the TensorProductElement
        return ufl.TensorProductElement(la, lb)
    else:
        return ufl.FiniteElement(family, cell=cell, degree=degree)


def VectorFunctionSpace(mesh, element, dim=None, name='fspace', si=0, parent=None):

    sub_element = make_scalar_element(element)
    dim = dim or mesh.ufl_cell().geometric_dimension()
    element = ufl.VectorElement(sub_element, dim=dim)
    return FunctionSpace(mesh, element, name=name)

# IMPLEMENT THIS!


def TensorFunctionSpace():
    pass
