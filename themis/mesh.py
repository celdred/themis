from petscshim import PETSc
import numpy as np
from functionspace import FunctionSpace
from function import Function
from bcs import DirichletBC
from solver import NonlinearVariationalProblem, NonlinearVariationalSolver
from ufl import inner, dx
from ufl_expr import TestFunction
from themiselement import ThemisElement
from ufl import Mesh, FiniteElement, interval, VectorElement, quadrilateral, TensorProductElement, hexahedron

# THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
decompfunction_code = r"""

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

from assembly import CompiledKernel
import ctypes

decompfunction = CompiledKernel(decompfunction_code, "decompfunction", cppargs=["-O3"], argtypes=[ctypes.c_voidp, ctypes.c_voidp,
                                                                                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ], restype=ctypes.c_int)


class SingleBlockMesh(Mesh):

    def create_dof_map(self, elem, ci):

        swidth = elem.swidth()
        ndof = elem.ndofs

        sizes = []
        ndofs_per_cell = []  # this is the number of "unique"/canonical dofs per element ie 1 for DG0, 2 for DG1, 1 for CG1, 2 for CG2, etc.
        for i in range(self.ndim):
            subelem = elem.sub_elements[ci][i]

            ndofs = subelem.ndofs_per_element
            nx = ndofs * self.nxs[i]
            if subelem.cont == 'H1' and self.bcs[i] == 'nonperiodic': nx = nx + 1
            ndofs_per_cell.append(ndofs)
            sizes.append(nx)

        da = PETSc.DMDA().create(dim=self.ndim, dof=ndof, proc_sizes=self.cell_da.getProcSizes(), sizes=sizes, boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=swidth, setup=False)

        # THIS IS AN UGLY HACK NEEDED BECAUSE OWNERSHIP RANGES ARGUMENT TO petc4py DMDA_CREATE is BROKEN
        bdx = list(np.array(sizes) - np.array(self.cell_da.getSizes(), dtype=np.int32) * np.array(ndofs_per_cell, dtype=np.int32))
        # bdx is the "extra" boundary dof for this space (0 if periodic, 1 if non-periodic)
        if self.ndim == 1:
            ndofs_per_cell.append(1)
            ndofs_per_cell.append(1)
            bdx.append(0)
            bdx.append(0)
        if self.ndim == 2:
            ndofs_per_cell.append(1)
            bdx.append(0)
        decompfunction([self.cell_da, da], [self.ndim, ndofs_per_cell[0], ndofs_per_cell[1], ndofs_per_cell[2], int(bdx[0]), int(bdx[1]), int(bdx[2])])
        da.setUp()

        return da


    def output(self, view):
        pass

    def destroy(self):
        self.cell_da.destroy()

    def get_boundary_direcs(self, subdomain):
        if subdomain == 'on_boundary':
            return self.bdirecs
        else:
            # ADD ERROR CHECKING HERE
            return self.bdirecs[subdomain]

    def __init__(self, nxs, bcs, name='singleblockmesh', coords=None, coordelem=None, procsizes=None):
        assert(len(nxs) == len(bcs))

        self.ndim = len(nxs)

        self.nxs = nxs
        self.bcs = bcs
        self._name = name
        self._blist = []
        self.bdirecs = []
        self.extruded = False

        for i, bc in enumerate(bcs):
            if bc == 'periodic':
                self._blist.append(PETSc.DM.BoundaryType.PERIODIC)
            else:
                self._blist.append(PETSc.DM.BoundaryType.NONE)  # GHOSTED
                # THIS WORKS FOR ALL THE UTILITY MESHES SO FAR
                # MIGHT NEED SOMETHING MORE GENERAL IN THE FUTURE?
                if i == 0:
                    direc = 'x'
                if i == 1:
                    direc = 'y'
                if i == 2:
                    direc = 'z'
                self.bdirecs.append(direc+'-')
                self.bdirecs.append(direc+'+')

        # generate mesh
        if not (procsizes is None):
            self.cell_da = PETSc.DMDA().create(dim=self.ndim, dof=1, sizes=nxs, boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=1, proc_sizes=procsizes)
        else:
            self.cell_da = PETSc.DMDA().create(dim=self.ndim, dof=1, sizes=nxs, boundary_type=self._blist, stencil_type=PETSc.DMDA.StencilType.BOX, stencil_width=1)

        if coords is None:
            if (coordelem is None):
                if len(nxs) == 1:
                    if bcs[0] == 'nonperiodic':
                        celem = FiniteElement("CG", interval, 1, variant='feec')
                    else:
                        celem = FiniteElement("DG", interval, 1, variant='feec')
                if len(nxs) == 2:
                    if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
                        celem = FiniteElement("Q", quadrilateral, 1, variant='feec')
                    else:
                        celem = FiniteElement("DQ", quadrilateral, 1, variant='feec')
                if len(nxs) == 3:
                    if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic' and bcs[2] == 'nonperiodic':
                        celem = FiniteElement("Q", hexahedron, 1, variant='feec')
                    else:
                        celem = FiniteElement("DQ", hexahedron, 1, variant='feec')
            else:
                # ADD SANITY CHECKS ON COORDINATE ELEMENT HERE
                celem = coordelem
        else:
            celem = coords.function_space()._uflelement.sub_elements()[0]

        vcelem = VectorElement(celem, dim=self.ndim)
        self._celem = celem
        self._vcelem = vcelem

        Mesh.__init__(self, self._vcelem)
        # print(celem,vcelem,coordelem,bcs)

        # construct and set coordsvec
        coordsspace = FunctionSpace(self, self._vcelem)
        self.coordinates = Function(coordsspace, name='coords')

        if coords is None:
            coordsdm = coordsspace.get_da(0)
            coordsarr = coordsdm.getVecArray(self.coordinates._vector)[:]
            newcoords = create_reference_domain_coordinates(self.cell_da, celem)
            if len(nxs) == 1:
                coordsarr[:] = np.squeeze(newcoords[:])
            else:
                coordsarr[:] = newcoords[:]
        else:
            coords._vector.copy(result=self.coordinates._vector)
        self.coordinates.scatter()


class SingleBlockExtrudedMesh(SingleBlockMesh):

    def __init__(self, basemesh, layers, layer_height = None, name='singleblockextrudedmesh', coords = None, vcoordelem=None):

        self.basemesh = basemesh
        basenxs = basemesh.nxs
        basebcs = basemesh.bcs

        basecoordelem = basemesh._celem

        if layer_height is None:
            layer_height = 1.0/layers

        if coords is None:
            if not (vcoordelem is None):
                # ADD SANITY CHECKS HERE
                velem = vcoordelem
            else:
                velem = FiniteElement("CG", interval, 1, variant='feec')

            celem = TensorProductElement(basecoordelem, velem)
        else:
            celem = coords.function_space()._uflelement.sub_elements()[0]
#################

        nxs = list(basenxs)
        bcs = list(basebcs)
        nxs.append(layers)
        bcs.append('nonperiodic')

        proc_sizes = list(basemesh.cell_da.getProcSizes())
        proc_sizes.append(1)
        SingleBlockMesh.__init__(self, nxs, bcs, name=name, coordelem=celem, procsizes=proc_sizes)

        # bdirecs shouldn't include the extruded directions...
        self.bdirecs = []
        for i, bc in enumerate(basebcs):
            if bc == 'nonperiodic':
                # THIS WORKS FOR ALL THE UTILITY MESHES SO FAR
                # MIGHT NEED SOMETHING MORE GENERAL IN THE FUTURE?
                if i == 0:
                    direc = 'x'
                if i == 1:
                    direc = 'y'
                self.bdirecs.append(direc+'-')
                self.bdirecs.append(direc+'+')

        self.extruded = True
        self.extrusion_dim = len(basenxs)

        # This is just uniform extrusion...
        if coords is None:
            basecoordsdm = basemesh.coordinates.space.get_da(0)
            basecoordsarr = basecoordsdm.getVecArray(basemesh.coordinates._vector)[:]

            coordsdm = self.coordinates.space.get_da(0)
            coordsarr = coordsdm.getVecArray(self.coordinates._vector)[:]

            if len(basecoordsarr.shape) == 1:
                basecoordsarr = np.expand_dims(basecoordsarr, axis=1)

            basecoordsarr = np.expand_dims(basecoordsarr, axis=-2)

            for i in range(len(basenxs)):
                coordsarr[..., i] = basecoordsarr[..., i]

            Lx = layer_height * layers
            coordsarr[..., -1] = coordsarr[..., -1] * Lx
        else:
            coords._vector.copy(result=self.coordinates._vector)
        self.coordinates.scatter()


    def get_boundary_direcs(self, subdomain):
        if subdomain == 'on_boundary':
            return self.bdirecs
        if subdomain == 'bottom':
            if self.extrusion_dim == 1:
                return ['y-', ]
            if self.extrusion_dim == 2:
                return ['z-', ]
        if subdomain == 'top':
            if self.extrusion_dim == 1:
                return ['y+', ]
            if self.extrusion_dim == 2:
                return ['z+', ]
        else:
            # ADD ERROR CHECKING HERE
            return self.bdirecs[subdomain]


# creates the reference domain [0,1]^d coordinate field
def create_reference_domain_coordinates(cell_da, celem):

    themis_celem = ThemisElement(celem)
    ranges = cell_da.getRanges()
    sizes = cell_da.getSizes()
    ndims = len(ranges)
    coordslist = []
    for i in range(ndims):
        subelem = themis_celem.sub_elements[0][i]
        nx = sizes[i]
        localnx = ranges[i][1]-ranges[i][0]
        dx = 1./nx
        ndofs = subelem.ndofs_per_element
        spts = subelem.spts[:ndofs]  # this makes CG skip the last dof, which is correct
        elementwisecoords = np.array(spts) * dx
        elementwisecoords = np.tile(elementwisecoords, localnx)
        elementcoords = np.linspace(ranges[i][0], ranges[i][1]-1, localnx) * dx  # this is left edge of each element
        elementcoords = np.repeat(elementcoords, ndofs)
        coords1D = elementcoords + elementwisecoords
        if subelem.cont == 'H1' and ranges[i][1] == sizes[i]:  # if this process owns the boundary
            coords1D = np.append(coords1D, 1.0)
        coordslist.append(coords1D)
    coordsND = np.meshgrid(*coordslist, indexing='ij')
    coordsND = np.stack(coordsND, axis=-1)

    return coordsND
