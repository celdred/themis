from ufl import FunctionSpace as UFLFunctionSpace
from ufl import MixedFunctionSpace as UFLMixedFunctionSpace
from petscshim import PETSc
from finiteelement import ThemisElement
import numpy as np


class FunctionSpace(UFLFunctionSpace):

    # returns the offset into a list of components for component ci
    # useful primarily for VectorSpaces, since they have multiple components
    def get_component_offset(self, ci):
        return self._component_offsets[ci]

    ################
    # returns the GLOBAL sizes for component ci
    def get_nxny(self, ci):
        sizes = self._da[ci].getSizes()
        return sizes

    # returns the LOCAL (WITHOUT GHOSTS) sizes for component ci
    def get_local_nxny(self, ci):
        ranges = self._da[ci].getRanges()
        nxs = []
        for xy in ranges:
            nxs.append((xy[1] - xy[0]))
        return nxs

    # returns the indices in global i/j/k space for component ci
    def get_xy(self, ci):
        ranges = self._da[ci].getRanges()
        return ranges
    # 3

    # returns the NON-GHOSTED total number of LOCAL dofs (ie per process) for component ci
    def get_localndofs(self, ci):
        ranges = self._da[ci].getRanges()
        nelem = 1
        for xy in ranges:
            nelem = (xy[1] - xy[0]) * nelem
        return nelem * self._da[ci].getDof()

    # returns the GHOSTED total number of LOCAL dofs (ie per process) for component ci
    def get_localghostedndofs(self, ci):
        ranges = self._da[ci].getGhostRanges()
        nelem = 1
        for xy in ranges:
            nelem = (xy[1] - xy[0]) * nelem
        return nelem * self._da[ci].getDof()

    def destroy(self):
        for ci in range(self.ncomp):
            self._da[ci].destroy()
            self._lgmaps[ci].destroy()
        self._composite_da.destroy()

    def output(self):
        pass
        # for ci in range(self.ncomp):
        #    for bi in range(self.npatches):
        #        da_output(self.get_da(ci, bi), self._name + '.da_cb' + str(ci) + '.' + str(bi))
        #        lgmap_output(self.get_lgmap(ci, bi), self._name + '.lgmap_cb.' + str(ci) + '.' + str(bi))
        #        is_output(self.get_component_block_lis(ci, bi), self._name + '.cb_lis.' + str(ci) + '.' + str(bi))
        # da_output(self._composite_da, self._name + '.compda')
        # lgmap_output(self.get_overall_lgmap(), self.name + '.overalllgmap')

    # DAs
    # return the da for component ci
    def get_da(self, ci):
        return self._da[ci]

    def get_composite_da(self):
        return self._composite_da

    # LGMAPS
    # returns the lgmap for component ci that goes from split local component indices to split component global indices
    def get_lgmap(self, ci):
        return self._lgmaps[ci]

    # returns the overal local to global map for the WHOLE SPACE
    def get_overall_lgmap(self):
        return self._overall_lgmap

    # returns the lgmap for component ci of space si that goes from split local to global (FOR THIS SPACE)
    # This is used exclusively for boundary conditions to transforms boundary indices from split local to global (either split-fieldwise or monolithic)
    def get_component_compositelgmap(self, si, ci):
        return self._component_lgmaps[self._component_offsets[ci]]

    # LIS
    # return the local IS for component ci - these are used in MatGetLocalSubMatrix
    # This is used to get the submatrix submatrix for component ci
    def get_component_block_lis(self, ci):
        return self._cb_lis[self._component_offsets[ci]]
        # return self.component_das[ci].getLocalISs()[bi]

    # DOES THIS WORK FOR VECTOR SPACES?
    # NO, IT DOESN'T TAKE INTO ACCOUNT THAT NDOF IS NOT EQUAL TO 1...
    # returns the SPLIT LOCAL indices of the owned portion of the physical boundary for component ci, block bi in direc
    def get_boundary_indices(self, ci, direc):

        # check if this space has any nodes on the boundary for ci in direc
        direcdict = {'x-': 0, 'x+': 0, 'y-': 1, 'y+': 1, 'z-': 2, 'z+': 2}
        intdirec = direcdict[direc]
        if self._themiselement.get_continuity(ci, intdirec) == 'L2':
            return [-1, ]

        da = self.get_da(ci)
        ndims = self._mesh.ndim
        nxs = self.get_nxny(ci)

        corners = da.getGhostCorners()
        ranges = da.getRanges()
        nx = corners[1][0]  # nxs[0]
        xs = corners[0][0]  # 0
        ix = np.array(ranges[0], dtype=np.int32) - xs
        jy = [0, 1]
        kz = [0, 1]
        ny = 1
        # nz = 1
        ys = 0
        zs = 0
        if ndims >= 2:
            ny = corners[1][1]  # nxs[1]
            ys = corners[0][1]  # 0
            jy = np.array(ranges[1], dtype=np.int32) - ys
        if ndims >= 3:
            # nz = corners[1][2]  # nxs[2]
            zs = corners[0][2]  # 0
            kz = np.array(ranges[2], dtype=np.int32) - zs

        if direc == 'x+':
            if ix[1] == nxs[0]-xs:
                ix = np.array([nxs[0]-1, nxs[0]], dtype=np.int32) - xs
            else:
                return [-1, ]
        if direc == 'x-':
            if ix[0] == 0-xs:
                ix = np.array([0, 1], dtype=np.int32) - xs
            else:
                return [-1, ]

        if direc == 'y+':
            if jy[1] == nxs[1]-ys:
                jy = np.array([nxs[1]-1, nxs[1]], dtype=np.int32) - ys
            else:
                return [-1, ]
        if direc == 'y-':
            if jy[0] == 0-ys:
                jy = np.array([0, 1], dtype=np.int32) - ys
            else:
                return [-1, ]

        if direc == 'z+':
            if kz[1] == nxs[2]-zs:
                kz = np.array([nxs[2]-1, nxs[2]], dtype=np.int32) - zs
            else:
                return [-1, ]
        if direc == 'z-':
            if kz[0] == 0-zs:
                kz = np.array([0, 1], dtype=np.int32) - zs
            else:
                return [-1, ]

        i1d = np.arange(ix[0], ix[1], dtype=np.int32)
        j1d = np.arange(jy[0], jy[1], dtype=np.int32)
        k1d = np.arange(kz[0], kz[1], dtype=np.int32)

        i, j, k = np.meshgrid(i1d, j1d, k1d, indexing='ij')
        rows = np.zeros(np.prod(i.shape), dtype=np.int32)
        i = np.ravel(i)
        j = np.ravel(j)
        k = np.ravel(k)
        rows[:] = i + nx*(j + k*ny)

        return rows

    #############################
    # These are included to simplify code: they allow FunctionSpace/VectorFunctionSpace to act as if they are a MixedFunctionSpace of size 1
    # returns the space for field si of a (mixed) space
    def get_space(self, si):
        return self

    # returns the offset for field si of a (mixed) space
    def get_space_offset(self, si):
        return 0

    # return the local field IS for field si- these are used for 2-form assemble
    # The None flag tells getblock that we should actually return the Matrix itself, not a submatrix
    def get_field_lis(self, si):
        return None

    #######################################################

    def mesh(self):
        return self.ufl_domain()

    def __len__(self):
        return self.ncomp

    def themis_element(self):
        return self._themiselement

    def split(self):
        """Split into a tuple of constituent spaces."""
        return (self, )

# HOW SHOULD OUTPUT/NAMES BE HANDLED HERE?
    def __init__(self, mesh, element, name='fspace', si=0, parent=None):

        self._themiselement = ThemisElement(element)
        self._uflelement = element
		
        self.ncomp = self._themiselement.get_ncomp()
        self._mesh = mesh
        self._name = name
        self._spaces = [self, ]
        self.nspaces = 1
		
        self._si = si
        if parent == None:
            self._parent = self
        else:
            self._parent = parent
		
        UFLFunctionSpace.__init__(self, mesh, element)

        # create das and lgmaps
        self._composite_da = PETSc.DMComposite().create()
        self._da = []
        self._lgmaps = []
        self._component_offsets = []
        for ci in range(self.ncomp):
            self._component_offsets.append(ci)
            da = mesh.create_dof_map(self._themiselement, ci)
            lgmap = da.getLGMap()
            self._composite_da.addDM(da)
            self._da.append(da)
            self._lgmaps.append(lgmap)
        self._composite_da.setUp()

        self._component_lgmaps = self._composite_da.getLGMaps()
        self._overall_lgmap = self._composite_da.getLGMap()
        self._cb_lis = self._composite_da.getLocalISs()
        
        #PETSc.Sys.Print(element,self._da[0][0].getGhostRanges(),self._da[0][0].getRanges())
        
class MixedFunctionSpace(UFLMixedFunctionSpace):

    def mesh(self):
        return self.ufl_domain()

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self.nspaces

    def sub(self,i):
        return self._spaces[i]
		
    def split(self):
        return self._spaces

    def __init__(self, spacelist,name='mixedspace'):
        #self._spaces = spacelist
        self.nspaces = len(spacelist)

        # ADD CHECK THAT ALL SPACES ARE DEFINED ON THE SAME MESH
        self._spaces = []
        for i,space in enumerate(spacelist):
            self._spaces.append(FunctionSpace(spacelist[i]._mesh,spacelist[i]._uflelement,name=name+str(i),si=i,parent=self))


        UFLMixedFunctionSpace.__init__(self, *spacelist)

        # create composite DM for component-block wise view
        self._composite_da = PETSc.DMComposite().create()
        for si in range(self.nspaces):
            for ci in range(self.get_space(si).ncomp):
                self._composite_da.addDM(self.get_space(si).get_da(ci))
        self._composite_da.setUp()

        # compute space offsets ie how far in a list of component until space k starts
        s = 0
        self._space_offsets = []
        for si in range(self.nspaces):
            self._space_offsets.append(s)
            s = s + self._spaces[si].ncomp

        # Create correct FIELD local and global IS's since DMComposite can't handle nested DMComposites in this case
        lndofs_total = 0
        for si in range(self.nspaces):  # determine offset into global vector
            for ci in range(self.get_space(si).ncomp):
                lndofs_total = lndofs_total + self.get_space(si).get_localndofs(ci)

        mpicomm = PETSc.COMM_WORLD.tompi4py()
        localcompoffset = mpicomm.scan(lndofs_total)
        localcompoffset = localcompoffset - lndofs_total  # This is the offset for this process

        self._field_lis = []
        self._field_gis = []
        ghostedlocaloffset = 0  # this is the offset for a given FIELD!
        localoffset = 0  # this is the offset for a given FIELD!
        for si in range(self.nspaces):
            # sum up the number of ghosted ndofs for the whole space
            totalghostedspacendofs = 0
            totalspacendofs = 0
            for ci in range(self.get_space(si).ncomp):
                totalghostedspacendofs = totalghostedspacendofs + self.get_space(si).get_localghostedndofs(ci)
                totalspacendofs = totalspacendofs + self.get_space(si).get_localndofs(ci)
            # create a strided index set of this size starting at the correct point
            self._field_lis.append(PETSc.IS().createStride(totalghostedspacendofs, first=ghostedlocaloffset, step=1, comm=PETSc.COMM_SELF))
            self._field_gis.append(PETSc.IS().createStride(totalspacendofs, first=localcompoffset + localoffset, step=1, comm=PETSc.COMM_WORLD))
            # adjust the FIELD offset
            ghostedlocaloffset = ghostedlocaloffset + totalghostedspacendofs
            localoffset = localoffset + totalspacendofs

        self._overall_lgmap = self._composite_da.getLGMap()
        self._component_lgmaps = self._composite_da.getLGMaps()

    # return the overall lgmap for the mixed space
    def get_overall_lgmap(self):
        return self._overall_lgmap

    def get_composite_da(self):
        return self._composite_da

    # returns the offset for field j of a (mixed) space in a list of components stored as si,ci,bi
    # access a specific component as soff+coff+bi, where soff is from get_space_offset and coff is from get_component_offset
    def get_space_offset(self, j):
        return self._space_offsets[j]

    # returns the space for field j of a (mixed) space
    def get_space(self, j):
        return self._spaces[j]

    # These operate on the split COMPONENT-WISE spaces
    # returns the lgmap for component ci, block bi of space si that goes from split local to global (FOR THIS SPACE)
    # This is used exclusively for boundary conditions to transforms boundary indices from split local to global (either split or monolithic)
    def get_component_compositelgmap(self, si, ci):
        soff = self._space_offsets[si]
        coff = self._spaces[si]._component_offsets[ci]
        return self._component_lgmaps[soff+coff]

    # These operate on the split FIELD-WISE spaces
    # returns the global IS for space si- these map from split global to monolithic global
    # They are used for setting fieldsplits and extracting subvectors
    def get_field_gis(self, si):
        return self._field_gis[si]  # self.composite_da_field.getGlobalISs()[si]

    # return the local IS for space si- these are used in MatGetLocalSubMatrix
    def get_field_lis(self, si):
        return self._field_lis[si]  # composite_da_field.getLocalISs()[si]


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
        
def VectorFunctionSpace(mesh, element, dim = None, name='fspace', si=0, parent=None):

    sub_element = make_scalar_element(element)
    dim = dim or mesh.ufl_cell().geometric_dimension()
    element = ufl.VectorElement(sub_element, dim=dim)
    return FunctionSpace(mesh, element, name=name)

#IMPLEMENT THIS!
def TensorFunctionSpace():
    pass
