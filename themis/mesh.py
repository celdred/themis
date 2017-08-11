from petscshim import PETSc
import instant
from compilation_options import *
import numpy as np
from functionspace import FunctionSpace
from function import Function

from ufl import Mesh, FiniteElement, interval, VectorElement, quadrilateral

# THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
decompfunction_code = r"""
#ifdef SWIG
%include "petsc4py/petsc4py.i"
#endif

#include <petsc.h>

PetscErrorCode decompfunction(DM cellda, DM sda, PetscInt dim, PetscInt cellndofsx, PetscInt cellndofsy, PetscInt cellndofsz, PetscInt bdx, PetscInt bdy, PetscInt bdz) { //,const PetscInt ly[], const PetscInt lz[]

PetscErrorCode ierr;
const PetscInt *lxcell,*lycell,*lzcell;
PetscInt *lx,*ly,*lz;
PetscInt pM,pN,pP;
PetscInt i;

ierr = DMDAGetInfo(cellda,0,0,0,0,&pM,&pN,&pP,0,0,0,0,0,0);
PetscMalloc1(pM,&lx);
PetscMalloc1(pN,&ly);
PetscMalloc1(pP,&lz);
ierr = DMDAGetOwnershipRanges(cellda,&lxcell,&lycell,&lzcell); CHKERRQ(ierr);
for (i=0;i<pM;i++)
{
lx[i] = lxcell[i] * cellndofsx;
}
lx[pM-1] = lx[pM-1] + bdx;
if (dim >= 2) {
for (i=0;i<pN;i++)
{
ly[i] = lycell[i] * cellndofsy;
}
ly[pN-1] = ly[pN-1] + bdy;
}
if (dim >= 3) {
for (i=0;i<pP;i++)
{
lz[i] = lzcell[i] * cellndofsz;
}
lz[pP-1] = lz[pP-1] + bdz;
}
//printf("%i %i %i %i %i\n",bdx,bdy,bdz,lxcell[0],lx[0]);
if (dim==1) {ierr = DMDASetOwnershipRanges(sda, lx, NULL, NULL); CHKERRQ(ierr);}
if (dim==2) {ierr = DMDASetOwnershipRanges(sda, lx, ly, NULL); CHKERRQ(ierr);}
if (dim==3) {ierr = DMDASetOwnershipRanges(sda, lx, ly, lz); CHKERRQ(ierr);}
PetscFree(lx);
PetscFree(ly);
PetscFree(lz);
return 0;
}
"""

decompfunction = instant.build_module(code=decompfunction_code,
                                      include_dirs=include_dirs,
                                      library_dirs=library_dirs,
                                      libraries=libraries,
                                      swig_include_dirs=swig_include_dirs).decompfunction


class MeshBase(Mesh):

    def create_dof_map(self, elem, ci, b):

        swidth = elem.maxdegree()
        ndof = elem.ndofs()

        sizes = []
        ndofs_per_cell = []  # this is the number of "unique" dofs per element ie 1 for DG0, 2 for DG1, 1 for CG1, 2 for CG2, etc.
        for i in range(self.ndim):
            nx = elem.get_nx(ci, i, self.nxs[b][i], self.bcs[i])
            ndofs = elem.get_ndofs_per_element(ci, i)

            ndofs_per_cell.append(ndofs)
            sizes.append(nx)

        da = PETSc.DMDA().create(dim=self.ndim, dof=ndof, proc_sizes=self._cell_das[b].getProcSizes(), sizes=sizes, boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=swidth, setup=False)

        # THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
        bdx = list(np.array(sizes) - np.array(self._cell_das[b].getSizes(), dtype=np.int32) * np.array(ndofs_per_cell, dtype=np.int32))
        # bdx is the "extra" boundary dof for this space (0 if periodic, 1 if non-periodic)
        if self.ndim == 1:
            ndofs_per_cell.append(1)
            ndofs_per_cell.append(1)
            bdx.append(0)
            bdx.append(0)
        if self.ndim == 2:
            ndofs_per_cell.append(1)
            bdx.append(0)
        decompfunction(self._cell_das[b], da, self.ndim, ndofs_per_cell[0], ndofs_per_cell[1], ndofs_per_cell[2], int(bdx[0]), int(bdx[1]), int(bdx[2]))
        da.setUp()

        return da

    def get_nxny(self, b):
        return self.nxs[b]

    def get_local_nxny_edge(self, edgeda):
        localnxs = []
        xys = edgeda.getRanges()
        for xy in xys:
            localnxs.append(xy[1]-xy[0])
        return localnxs

    def get_local_nxny(self, b):
        localnxs = []
        xys = self.get_xy(b)
        for xy in xys:
            localnxs.append(xy[1]-xy[0])
        return localnxs

    def get_xy(self, b):
        return self._cell_das[b].getRanges()

    def get_cell_da(self, b):
        return self._cell_das[b]

    def output(self, view):
        for b in range(self._npatches):
            da_output(self._cell_das[b], self.name + '.da.'+str(b))
            da_output(self._edgex_das[b], self.name + '.edgex.da.'+str(b))
            if self.ndim >= 2:
                da_output(self._edgey_das[b], self.name + '.edgey.da.'+str(b))
            if self.ndim >= 3:
                da_output(self._edgez_das[b], self.name + '.edgez.da.'+str(b))

    def destroy(self):
        for b in range(self.npatches):
            self._cell_das[b].destroy()
            self._edgex_das[b].destroy()
            if self.ndim >= 2:
                self._edgey_das[b].destroy()
            if self.ndim >= 3:
                self._edgez_das[b].destroy()


# HOW SHOULD OUTPUT/NAMES BE HANDLED HERE?
class SingleBlockMesh(MeshBase):

    def __init__(self, nxs, bcs, name='singleblockmesh', coordelem=None):
        assert(len(nxs) == len(bcs))

        self.ndim = len(nxs)

        self.nxs = [nxs, ]
        self.bcs = bcs
        self.npatches = 1
        self.patchlist = []
        self._name = name
        self._blist = []

        for bc in bcs:
            if bc == 'periodic':
                self._blist.append(PETSc.DM.BoundaryType.PERIODIC)
            else:
                self._blist.append(PETSc.DM.BoundaryType.GHOSTED)

        bdx = 0
        bdy = 0
        bdz = 0
        edgex_nxs = list(nxs)
        if (bcs[0] == 'nonperiodic'):
            edgex_nxs[0] = edgex_nxs[0] + 1
            bdx = 1
        if self.ndim >= 2:
            edgey_nxs = list(nxs)
            if (bcs[1] == 'nonperiodic'):
                edgey_nxs[1] = edgey_nxs[1] + 1
                bdy = 1

        if self.ndim >= 3:
            edgez_nxs = list(nxs)
            if (bcs[2] == 'nonperiodic'):
                edgez_nxs[2] = edgez_nxs[2] + 1
                bdz = 1

        # generate mesh
        cell_da = PETSc.DMDA().create()
        # THIS SHOULD REALLY BE AN OPTIONS PREFIX THING TO MESH...
        # MAIN THING IS THE ABILITY TO CONTROL DECOMPOSITION AT COMMAND LINE
        # BUT WE WANT TO BE ABLE TO IGNORE DECOMPOSITION WHEN RUNNING WITH A SINGLE PROCESSOR
        # WHICH ALLOWS THE SAME OPTIONS LIST TO BE USED TO RUN SOMETHING IN PARALLEL, AND THEN DO PLOTTING IN SERIAL!
        if PETSc.COMM_WORLD.size > 1:  # this allows the same options list to be used to run something in parallel, but then plot it in serial!
            cell_da.setOptionsPrefix(self._name + '0_')
            cell_da.setFromOptions()
        #############

        cell_da.setDim(self.ndim)
        cell_da.setDof(1)
        cell_da.setSizes(nxs)
        cell_da.setBoundaryType(self._blist)
        cell_da.setStencil(PETSc.DMDA.StencilType.BOX, 1)
        cell_da.setUp()

        # print self.cell_da.getProcSizes()
        edgex_da = PETSc.DMDA().create(dim=self.ndim, dof=1, sizes=edgex_nxs, proc_sizes=cell_da.getProcSizes(), boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=1, setup=False)
        # THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
        decompfunction(cell_da, edgex_da, self.ndim, 1, 1, 1, bdx, 0, 0)
        edgex_da.setUp()

        if self.ndim >= 2:
            edgey_da = PETSc.DMDA().create(dim=self.ndim, dof=1, sizes=edgey_nxs, proc_sizes=cell_da.getProcSizes(), boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=1, setup=False)
            # THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
            decompfunction(cell_da, edgey_da, self.ndim, 1, 1, 1, 0, bdy, 0)
            edgey_da.setUp()
        if self.ndim >= 3:
            edgez_da = PETSc.DMDA().create(dim=self.ndim, dof=1, sizes=edgez_nxs, proc_sizes=cell_da.getProcSizes(), boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=1, setup=False)
            # THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
            decompfunction(cell_da, edgez_da, self.ndim, 1, 1, 1, 0, 0, bdz)
            edgez_da.setUp()

        self._cell_das = [cell_da, ]
        self._edgex_das = [edgex_da, ]
        self._edgex_nxs = [edgex_nxs, ]
        if self.ndim >= 2:
            self._edgey_das = [edgey_da, ]
            self._edgey_nxs = [edgey_nxs, ]
        if self.ndim >= 3:
            self._edgez_das = [edgez_da, ]
            self._edgez_nxs = [edgez_nxs, ]

        if not (coordelem is None):
            celem = coordelem
        else:
            if len(nxs) == 1:
                # celem = FiniteElement("DG",interval,1) #FIX THIS- HOW SHOULD WE DEFAULT HERE?
                celem = FiniteElement("DG", interval, 1, variant='feec')  # FIX THIS- HOW SHOULD WE DEFAULT HERE?
            if len(nxs) == 2:
                # celem = FiniteElement("DG",quadrilateral,1) #FIX THIS- HOW SHOULD WE DEFAULT HERE?
                celem = FiniteElement("DG", quadrilateral, 1, variant='feec')  # FIX THIS- HOW SHOULD WE DEFAULT HERE?
            if len(nxs) == 3:
                raise ValueError("3D NEEDS TO BE A TENSOR PRODUCT ELEMENT- FIX THIS!")
        celem = VectorElement(celem, dim=self.ndim)

        Mesh.__init__(self, celem)

        # THIS BREAKS FOR COORDELEM NOT DG1...
        # construct and set coordsvec
        coordsspace = FunctionSpace(self, celem)
        self.coordinates = Function(coordsspace, name='coords')

        localnxs = self.get_local_nxny(0)
        newcoords = create_reference_domain_coordinates(self.get_cell_da(0), nxs, bcs, localnxs)

        coordsdm = coordsspace.get_da(0, 0)
        coordsarr = coordsdm.getVecArray(self.coordinates._vector)[:]

        if len(nxs) == 1:
            coordsarr[:] = np.squeeze(newcoords[:])
        else:
            coordsarr[:] = newcoords[:]

        self.coordinates.scatter()

# creates the reference domain [0,nx]^d coordinate field


def create_reference_domain_coordinates(dm, nxs, bcs, localnxs):
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0
    zmin = 0.0
    zmax = 0.0
    # This is take care of the fact that DmDA set uniform coordinates uses vertex-centered coordinates, so it divides by nx-1 in the non-periodic case
    if len(nxs) >= 1:
        if bcs[0] == 'periodic':
            xmax = nxs[0]
        if bcs[0] == 'nonperiodic':
            xmax = nxs[0] - 1
    if len(nxs) >= 2:
        if bcs[1] == 'periodic':
            ymax = nxs[1]
        if bcs[1] == 'nonperiodic':
            ymax = nxs[1] - 1
    if len(nxs) >= 3:
        if bcs[2] == 'periodic':
            zmax = nxs[2]
        if bcs[2] == 'nonperiodic':
            zmax = nxs[2] - 1
    dm.setUniformCoordinates(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
    coordsvec = dm.getCoordinates()
    coordsdm = dm.getCoordinateDM()
    coordsarr = coordsdm.getVecArray(coordsvec)[:]

    coordsarr = coordsarr + 0.5

    coordslist = []
    expandedcoords = np.array(coordsarr)
    for i in range(len(nxs)):
        # if bcs[i] == 'periodic':
        coords1D = np.tile([-0.5, 0.5], localnxs[i])  # This is DG1 ie 2 dofs per element
        expandedcoords = np.repeat(expandedcoords, 2, axis=i)
        # if bcs[i] == 'nonperiodic':
        # coords1D = np.tile([-0.5*dxs[i]],nxs[i]+1) #This is CG1 ie 1 dof per element
        # BROKEN- NEED TO PROPERLY SET EXPANDED COORDS
        # JUST ADD ONE EXTRA ELEMENT...
        # BUT ONLY AT THE TRUE END OF THE ARRAY...
        # STRONG ALTERNATIVE TO ALL THIS BULLSHIT IS TO ALWAYS TREAT COORDS LIKE A DG1 SPACE...THIS IS A LOT SIMPLER AND IMMEDIATELY WORKS FOR PERIODIC, ALSO FOR HEDGEHOG STYLE
        # ADDS A LITTLE BIT OF EXPENSE AND EXTRA MEMORY, BUT WHATEVER!
        coordslist.append(coords1D)
    coordsND = np.array(np.meshgrid(*coordslist, indexing='ij'))
    if len(nxs) == 1:
        expandedcoords = np.expand_dims(expandedcoords, axis=1)
    revisedcoordsND = np.zeros(expandedcoords.shape)
    for i in range(len(nxs)):
        revisedcoordsND[..., i] = coordsND[i, ...] + expandedcoords[..., i]

    return revisedcoordsND
