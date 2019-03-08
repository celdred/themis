from petscshim import PETSc
# import instant
# from compilation_options import *
import numpy as np
from functionspace import FunctionSpace
from function import Function
from bcs import DirichletBC
from solver import NonlinearVariationalProblem, NonlinearVariationalSolver
from ufl import inner, dx
from ufl_expr import TestFunction
from themiselement import ThemisElement
from ufl import Mesh, FiniteElement, interval, VectorElement, quadrilateral, TensorProductElement, hexahedron

class SingleBlockMesh(Mesh):

    def create_dof_map(self, elem):

        swidth = elem.swidth
        
        dofmap = elem.dofmap
        ndofs = elem.ndofs
        
        da = PETSc.DMStag.create(self, self.ndim, dofmap*ndofs, self.nxs, self._blist, PETSc.DMStag.StencilType.BOX, swidth)
        
        return da

    def get_da(self):
        return self.da

    def output(self, view):
        pass

    def destroy(self):
        self.da.destroy()
        
    def get_boundary_direcs(self, subdomain):
        if subdomain == 'on_boundary':
            return self.bdirecs
        else:
            # ADD ERROR CHECKING HERE
            return self.bdirecs[subdomain]

    def boundary_element_indices(self,direc):
        return self.boundary_indices[direc]
        
    # PRECOMPUTE WHEN MESH IS CREATED BASED ON BCS...
    
    def __init__(self, nxs, bcs, name='singleblockmesh'):
        assert(len(nxs) == len(bcs))

        self.ndim = len(nxs)

        self.nxs = nxs
        self.bcs = bcs
        self._name = name
        self.extruded = False
        
        bstring_to_btype = {'periodic': PETSc.DM.BoundaryType.PERIODIC, 'nonperiodic': PETSc.DM.BoundaryType.NONE}
        self._blist = [bstring_to_btype[bstring] for bstring in bcs]
        
        # generate mesh
        if self.ndim == 1: dof = (0,1)
        if self.ndim == 2: dof = (0,0,1)
        if self.ndim == 3: dof = (0,0,0,1)
        self.da = PETSc.DMStag.create(self, self.ndim, dof, nxs, self._blist, PETSc.DMStag.StencilType.NONE, 0)

        # set initial (=uniform) coordinates
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

        vcelem = VectorElement(celem, dim=self.ndim)
        self._celem = celem
        self._vcelem = vcelem
        Mesh.__init__(self, vcelem)
        # print(celem,vcelem,coordelem,bcs)

        # construct and set coordsvec
        coordsspace = FunctionSpace(self, vcelem)
        self.coordinates = Function(coordsspace, name='coords')
        coordsdm = coordsspace.get_da()
        coordsarr = coordsdm.getVecArray(self.coordinates._lvectors[0])[:]
        newcoords = create_reference_domain_coordinates(da, self.ndim, celem)
        if len(nxs) == 1:
            coordsarr[:] = np.squeeze(newcoords[:])
        else:
            coordsarr[:] = newcoords[:]
        self.da.localToGlobal(self.coordinates._lvectors[0],self.coordinates._vector,addv=PETSc.InsertMode.INSERT)
        
        # set boundary indices and bdirecs
        self.bdirecs = []
        self.boundary_indices = {}
        direcdict = {0: 'x', 1: 'y', 2: 'z'}
        firstrank = self.da.isFirstRank()
        lastrank = self.da.isLastRank()
        starts,widths,extra = self.da.getCorners()
        for i, bc in enumerate(bcs):
            if bc == 'periodic':
                # there are no boundaries if this direction is periodic
                self.boundary_indices[direcdict[i]+'-'] = (None,None)
                self.boundary_indices[direcdict[i]+'+'] = (None,None)
            else:
                # THIS WORKS FOR ALL THE UTILITY MESHES SO FAR
                # MIGHT NEED SOMETHING MORE GENERAL IN THE FUTURE?
                self.bdirecs.append(direcdict[i]+'-')
                self.bdirecs.append(direcdict[i]+'+')

                # check is we own the boundary in the relevant direction, and if so, compute it
                # default to looping over the whole owned set of elements
                # this works for ndim < 3 because getCorners defaults to returning xs=0, xm=1, nextra = 0 for "unused" dimensions 
                if firstrank[i]: 
                    xm = [widths[0] + extra[0],widths[1] + extra[1],widths[2] + extra[2]]
                    xs = [starts[0],starts[1],starts[2]]
                    xs[i] = starts[i]
                    xm[i] = 1
                    self.boundary_indices[direcdict[i]+'-'] = (xs,xm)
                else:
                    self.boundary_indices[direcdict[i]+'-'] = (None,None)
                
                if lastrank[i]:
                    xm = [widths[0] + extra[0],widths[1] + extra[1],widths[2] + extra[2]]
                    xs = [starts[0],starts[1],starts[2]]
                    xs[i] = starts[i] + widths[i] + extra[i]
                    xm[i] = 1               
                    self.boundary_indices[direcdict[i]+'+'] = (xs,xm)
                else:
                    self.boundary_indices[direcdict[i]+'+'] = (None,None)
            
# creates the reference domain [0,1]^d coordinate field
def create_reference_domain_coordinates(da, ndims, celem):

    themis_celem = ThemisElement(celem)
    xs,xm = da.getCorners()
    gs = da.getGlobalSizes()
    coordslist = []
    lastrank = da.isLastRank()

# HERE WE CAN SPECIALIZE TO EITHER CG1 or DG1!
# SO CONTINUITY CAN BASICALLY GO AWAY, CAN JUST LOOK AT ELEMNAME
    for i in range(ndims):
        nx = gs[i]
        dx = 1./nx
        if themis_celem.get_continuity(0, i) == 'H1':
            spts = themis_celem._spts[0][i][:-1] # skip the last dof
        else:
            spts = themis_celem._spts[0][i][:]
        elementwisecoords = np.array(spts) * dx
        elementwisecoords = np.tile(elementwisecoords, xm[i])
        elementcoords = np.linspace(xs[i], xs[i]+xm[i]-1, xm[i]) * dx  # this is left edge of each element
        elementcoords = np.repeat(elementcoords, ndofs)
        coords1D = elementcoords + elementwisecoords
        if themis_celem.get_continuity(0, i) == 'H1' and lastrank[i]:  # if this process owns the boundary
            coords1D = np.append(coords1D, 1.0)
        coordslist.append(coords1D)
        
    coordsND = np.meshgrid(*coordslist, indexing='ij')
    coordsND = np.stack(coordsND, axis=-1)

    return coordsND
    
    

# THIS IS BROKEN- SHOULD REALLY USE A DMStagExtruded Object...

class SingleBlockExtrudedMesh(SingleBlockMesh):

    def __init__(self, basemesh, layers, layer_height, name='singleblockextrudedmesh', vcoordelem=None):

        self.basemesh = basemesh
        basenxs = basemesh.nxs
        basebcs = basemesh.bcs

        basecoordelem = basemesh._celem
        if not (vcoordelem is None):
            # ADD SANITY CHECKS HERE
            velem = vcoordelem
        else:
            velem = FiniteElement("CG", interval, 1, variant='feec')

        celem = TensorProductElement(basecoordelem, velem)
#################

        nxs = list(basenxs)
        bcs = list(basebcs)
        nxs.append(layers)
        bcs.append('nonperiodic')

        proc_sizes = list(basemesh.da.getProcSizes())
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
        basecoordsdm = basemesh.coordinates.space.get_da()

        basecoordsarr = basecoordsdm.getVecArray(basemesh.coordinates._vector)[:]

        coordsdm = self.coordinates.space.get_da()
        coordsarr = coordsdm.getVecArray(self.coordinates._vector)[:]

        if len(basecoordsarr.shape) == 1:
            basecoordsarr = np.expand_dims(basecoordsarr, axis=1)

        basecoordsarr = np.expand_dims(basecoordsarr, axis=-2)

        for i in range(len(basenxs)):
            coordsarr[..., i] = basecoordsarr[..., i]

        Lx = layer_height * layers
        coordsarr[..., -1] = coordsarr[..., -1] * Lx

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



