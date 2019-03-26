
from interop import FiniteElement, TensorProductElement, VectorElement
from interop import quadrilateral, interval, triangle, hexahedron
from interop import Function, FunctionSpace
from interop import IntervalMesh, PeriodicIntervalMesh
from interop import SquareMesh, PeriodicSquareMesh, RectangleMesh, PeriodicRectangleMesh
from interop import ExtrudedMesh, Mesh, BoxMesh
from interop import CubedSphereMesh, IcosahedralSphereMesh
from interop import SpatialCoordinate
from interop import coordvariant

__all__ = ['create_box_mesh', 'set_mesh_quadrature', 'set_boxmesh_operators']

def set_mesh_coordinate_order(mesh, cell, bcs, coordorder, vcoordorder=None):
    vcoordorder = vcoordorder or coordorder

    if cell == 'interval':
        if bcs[0] == 'nonperiodic':
            celem = FiniteElement("CG", interval, coordorder, variant=coordvariant)
        else:
            celem = FiniteElement("DG", interval, coordorder, variant=coordvariant)

    if cell == 'quad':
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            celem = FiniteElement("Q", quadrilateral, coordorder, variant=coordvariant)
        else:
            celem = FiniteElement("DQ", quadrilateral, coordorder, variant=coordvariant)

    if cell == 'tri':
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            celem = FiniteElement("CG", triangle, coordorder, variant=coordvariant)
        else:
            celem = FiniteElement("DG", triangle, coordorder, variant=coordvariant)

    if cell == 'tpquad':
        if bcs[0] == 'nonperiodic':
            hcelem = FiniteElement("CG", interval, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        else:
            hcelem = FiniteElement("DG", interval, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        celem = TensorProductElement(hcelem, vcelem)

    if cell == 'hex':
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic' and bcs[2] == 'nonperiodic':
            celem = FiniteElement("Q", hexahedron, coordorder, variant=coordvariant)
        else:
            celem = FiniteElement("DQ", hexahedron, coordorder, variant=coordvariant)

    if cell == 'tphex':
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            hcelem = FiniteElement("Q", quadrilateral, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        else:
            hcelem = FiniteElement("DQ", quadrilateral, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        celem = TensorProductElement(hcelem, vcelem)

    if cell == 'tptri':
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            hcelem = FiniteElement("CG", triangle, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        else:
            hcelem = FiniteElement("DG", triangle, coordorder, variant=coordvariant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant=coordvariant)
        celem = TensorProductElement(hcelem, vcelem)

    velem = VectorElement(celem)

    coordspace = FunctionSpace(mesh, velem)
    newcoords = Function(coordspace)
    newcoords.interpolate(SpatialCoordinate(mesh))
    return Mesh(newcoords)


def create_box_mesh(cell, nxs, xbcs, lxs, coordorder, vcoordorder=None):

    if cell in ['quad', 'tphex']:
        use_quad = True
    if cell in ['tri', 'tptri']:
        use_quad = False

    if cell == 'interval':
        if xbcs[0] == 'nonperiodic':
            mesh = IntervalMesh(nxs[0], lxs[0])
        if xbcs[0] == 'periodic':
            mesh = PeriodicIntervalMesh(nxs[0], lxs[0])

    if cell in ['tri', 'quad']:
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            mesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            mesh = RectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            mesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            mesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], direction='y', quadrilateral=use_quad)

    if cell == 'tpquad':
        if xbcs[1] == 'periodic':
            raise ValueError('cannot make an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'nonperiodic':
            bmesh = IntervalMesh(nxs[0], lxs[0])
        if xbcs[0] == 'periodic':
            bmesh = PeriodicIntervalMesh(nxs[0], lxs[0])
        mesh = ExtrudedMesh(bmesh, nxs[1], layer_height=lxs[1]/nxs[1])

# CURRENTLY BROKEN FOR FIREDRAKE
    if cell == 'hex':
        mesh = BoxMesh(nxs[0], nxs[1], nxs[2], lxs[0], lxs[1], lxs[2], xbcs[0], xbcs[1], xbcs[2])

    if cell in ['tphex', 'tptri']:
        if xbcs[2] == 'periodic':
            raise ValueError('cannot make an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            bmesh = RectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            bmesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicRectangleMesh(nxs[0], nxs[1], lxs[0], lxs[1], direction='y', quadrilateral=use_quad)
        mesh = ExtrudedMesh(bmesh, nxs[2], layer_height=lxs[2]/nxs[2])

    mesh.extruded = False
    if cell in ['tpquad', 'tphex', 'tptri']:
        mesh.extruded = True

    # Upgrade coordinate order if needed
    if coordorder > 1:
        newmesh = set_mesh_coordinate_order(mesh, cell, xbcs, coordorder, vcoordorder=vcoordorder)
    else:
        newmesh = mesh

    return newmesh

# FIX THIS
# def create_sphere_mesh(cell, rlevel, radius, coordorder, Lz=1.0, nlevels=None, extrusion_type='uniform', vcoordorder=None):
#
#     if cell == 'quad':
#         mesh = CubedSphereMesh(radius, refinement_level=rlevel, degree=coordorder)
#     if cell == 'tri':
#         mesh = IcosahedralSphereMesh(radius, refinement_level=rlevel, degree=coordorder)
#     if cell == 'tphex':
#         basemesh = CubedSphereMesh(radius, refinement_level=rlevel, degree=coordorder)
#         mesh = ExtrudedMesh(bmesh, nlevels, layer_height=Lz/nlevels, extrusion_type=extrusion_type)
#     if cell == 'tptri':
#         basemesh = IcosahedralSphereMesh(radius, refinement_level=rlevel, degree=coordorder)
#         mesh = ExtrudedMesh(bmesh, nlevels, layer_height=Lz/nlevels, extrusion_type=extrusion_type)
#
#     # Upgrade coordinate order if needed
# # THIS IS BROKEN
# # HOW ARE COORDINATES TREATED FOR SPHERICAL MESHES ANYWAYS
# # DEPENDS ON EXTRUSION TYPE...
#     if coordorder > 1:
#         newmesh = set_mesh_coordinate_order(
#             mesh, ndims, xbcs, coordorder, vcoordorder=vcoordorder)
#     else:
#         newmesh = mesh
#
#     newmesh.cell = cell
#
#     return newmesh




def set_mesh_quadrature(mesh, cell, qdegree=None, vqdegree=None):

    if qdegree:
        if cell in ['interval', 'quad', 'tri', 'hex']:
            mesh.dx = dx(metadata={"quadrature_degree": qdegree})
            mesh.dS = dS(metadata={"quadrature_degree": qdegree})
            mesh.ds = ds(metadata={"quadrature_degree": qdegree})
# EVENTUALLY THIS SHOULD ACTUALLY DO SEPARATE DEGREES IN HORIZ AND VERTICAL...
        if cell in ['tpquad', 'tphex', 'tptri']:
            vqdegree = vqdegree or qdegree
            # THIS SHOULD BE FIXED AT SOME POINT...
            qdegree = max(qdegree, vqdegree)
            mesh.dx = dx(metadata={"quadrature_degree": qdegree})
            mesh.ds_b = ds_b(metadata={"quadrature_degree": qdegree})
            mesh.ds_t = ds_t(metadata={"quadrature_degree": qdegree})
            mesh.ds_v = ds_v(metadata={"quadrature_degree": qdegree})
            mesh.dS_h = dS_h(metadata={"quadrature_degree": qdegree})
            mesh.dS_v = dS_v(metadata={"quadrature_degree": qdegree})
            mesh.dS = mesh.dS_h + mesh.dS_v
            mesh.ds_tb = mesh.ds_t + mesh.ds_b
            mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v
    else:
        if cell in ['interval', 'quad', 'tri', 'hex']:
            mesh.dx = dx
            mesh.dS = dS
            mesh.ds = ds
        if cell in ['tpquad', 'tphex', 'tptri']:
            mesh.dx = dx
            mesh.ds_b = ds_b
            mesh.ds_v = ds_v
            mesh.ds_t = ds_t
            mesh.dS_h = dS_h
            mesh.dS_v = dS_v
            mesh.dS = mesh.dS_h + mesh.dS_v
            mesh.ds_tb = mesh.ds_t + mesh.ds_b
            mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v


def set_boxmesh_operators(mesh, cell):

# ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# Basically what we need to make Hamiltonian and Hodge Laplacian stuff dimension-agnostic
    if cell in ['quad', 'tri', 'tpquad']:
            mesh.skewgrad = lambda s: perp(grad(s))
            mesh.perp = lambda u: perp(u)

    if cell in ['tpquad']:
        mesh.k = as_vector((0, 1))

    if cell in ['hex', 'tphex', 'tptri']:
        #mesh.skewgrad = lambda s : cross(mesh.k,grad(s))
        #mesh.perp = lambda u : cross(mesh.k,u)
        pass

        if cell in ['tphex', 'tptri']:
            mesh.k = as_vector((0, 0, 1))


def make_sphere_normals(x):
    R = sqrt(inner(x, x))
    rhat_expr = x/R
    return rhat_expr

# ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# def set_spheremesh_operators(mesh, cell):
#
#     if cell in ['quad', 'tri']:
#         xs = SpatialCoordinate(mesh)
#         mesh.cell_normals = make_sphere_normals(xs)
#         mesh.perp = lambda u: cross(mesh.cell_normals, u)
#         mesh.skewgrad = lambda s: cross(mesh.cell_normals, grad(s))
#         global_normal = as_vector((xs[0], xs[1], xs[2]))
#         mesh.init_cell_orientations(global_normal)
# # FIX
#     if cell in ['tphex', 'tptri']:
#         pass
