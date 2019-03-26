
from interop import FiniteElement, TensorProductElement, VectorElement
from interop import quadrilateral, interval, triangle, hexahedron
from interop import Function, FunctionSpace
from interop import IntervalMesh, PeriodicIntervalMesh
from interop import SquareMesh, PeriodicSquareMesh, RectangleMesh, PeriodicRectangleMesh
from interop import ExtrudedMesh, Mesh, CubeMesh, BoxMesh
from interop import CubedSphereMesh, IcosahedralSphereMesh
from interop import SpatialCoordinate

# THIS ALSO FAILS HARD FOR TRIANGULAR MESHES!
def set_mesh_coordinate_order(mesh, ndims, bcs, coordorder, vcoordorder=None):
    vcoordorder = vcoordorder or coordorder

# EXTRUSION CHECKS MIGHT FAIL FOR FIREDRAKE...
# ALSO HOW DO WE PROPERLY HANDLE VARIANT...
# WE NEED THE VARIANT FLAG FOR THEMIS BUT IT WILL FAIL FOR FIREDRAKE!
# Basically something needs to automatically convert variant 'feec' -> None
    variant = 'feec'
# FIX THIS TO SUPPORT BOTH FIREDRAKE AND THEMIS

    if ndims == 1:
        if bcs[0] == 'nonperiodic':
            celem = FiniteElement("CG", interval, coordorder, variant = variant)
        else:
            celem = FiniteElement("DG", interval, coordorder, variant = variant)
    if ndims == 2 and not mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            celem = FiniteElement("Q", quadrilateral, coordorder, variant = variant)
        else:
            celem = FiniteElement("DQ", quadrilateral, coordorder, variant = variant)
    if ndims == 2 and mesh.extruded:
        if bcs[0] == 'nonperiodic':
            hcelem = FiniteElement("CG", interval, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        else:
            hcelem = FiniteElement("DG", interval, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        celem = TensorProductElement(hcelem,vcelem)
    if ndims == 3 and not mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic' and bcs[2] == 'nonperiodic':
            celem = FiniteElement("Q", hexahedron, coordorder, variant = variant)
        else:
            celem = FiniteElement("DQ", hexahedron, coordorder, variant = variant)
    if ndims == 3 and mesh.extruded:
        if bcs[0] == 'nonperiodic' and bcs[1] == 'nonperiodic':
            hcelem = FiniteElement("Q", quadrilateral, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        else:
            hcelem = FiniteElement("DQ", quadrilateral, coordorder, variant = variant)
            vcelem = FiniteElement("CG", interval, vcoordorder, variant = variant)
        celem = TensorProductElement(hcelem,vcelem)

    velem = VectorElement(celem, dim=ndims)
    coordspace = FunctionSpace(mesh, velem)
    newcoords = Function(coordspace)
    newcoords.interpolate(SpatialCoordinate(mesh))
    return Mesh(newcoords)

def create_box_mesh(cell, nxs, xbcs, coordorder, vcoordorder = None):

    if cell in ['quad', 'tphex']:
        use_quad = True
    if cell in ['tri', 'tptri']:
        use_quad = False

    if cell == 'interval':
        ndims = 1
        if xbcs[0] == 'nonperiodic':
            mesh = IntervalMesh(nxs[0], 1.0)
        if xbcs[0] == 'periodic':
            mesh = PeriodicIntervalMesh(nxs[0], 1.0)

    if cell in ['tri', 'quad']:
        ndims = 2
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            mesh = SquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            mesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='y', quadrilateral=use_quad)
    if cell == 'tpquad':
        ndims = 2
        if xbcs[1] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'nonperiodic':
            bmesh = IntervalMesh(nxs[0], 1.0)
        if xbcs[0] == 'periodic':
            bmesh = PeriodicIntervalMesh(nxs[0], 1.0)
        mesh = ExtrudedMesh(bmesh, nxs[1])

    if cell == 'hex':
        ndims = 3
        mesh = CubeMesh(nxs[0], nxs[1], nxs[2], 1.0, xbcs[0], xbcs[1], xbcs[2])
    if cell in ['tphex', 'tptri']:
        ndims = 3
        if xbcs[2] == 'periodic':
            raise ValueError('cannot do an extruded mesh with periodic bcs in direction of extrusion')
        if xbcs[0] == 'periodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'nonperiodic':
            bmesh = SquareMesh(nxs[0], nxs[1], 1.0, quadrilateral=use_quad)
        if xbcs[0] == 'periodic' and xbcs[1] == 'nonperiodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='x', quadrilateral=use_quad)
        if xbcs[0] == 'nonperiodic' and xbcs[1] == 'periodic':
            bmesh = PeriodicSquareMesh(nxs[0], nxs[1], 1.0, direction='y', quadrilateral=use_quad)
        mesh = ExtrudedMesh(bmesh, nxs[2])

    # Required since Firedrake doesn't flag meshes as extruded
    mesh.extruded = False
    if cell in ['tpquad', 'tphex', 'tptri']: mesh.extruded = True

    # Upgrade coordinate order if needed
    if coordorder > 1:
        newmesh = set_mesh_coordinate_order(mesh, ndims, xbcs, coordorder, vcoordorder=vcoordorder)
    else:
        newmesh = mesh

    return newmesh

# FIX THIS
def create_sphere_mesh(cell, rlevel, coordorder, nlevels = None, extrusion_type='uniform', vcoordorder = None):

    if cell == 'quad':
        mesh = CubedSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
    if cell == 'tri':
        mesh = IcosahedralSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
    if cell == 'tphex':
        basemesh = CubedSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
        mesh = ExtrudedMesh(basemesh, nlevels, extrusion_type=extrusion_type)
    if cell == 'tptri':
        basemesh = IcosahedralSphereMesh(1.0, refinement_level=rlevel, degree=coordorder)
        mesh = ExtrudedMesh(basemesh, nlevels, extrusion_type=extrusion_type)

    # Upgrade coordinate order if needed
# THIS IS BROKEN
# HOW ARE COORDINATES TREATED FOR SPHERICAL MESHES ANYWAYS
# DEPENDS ON EXTRUSION TYPE...
    if coordorder > 1:
        newmesh = set_mesh_coordinate_order(mesh, ndims, xbcs, coordorder, vcoordorder=vcoordorder)
    else:
        newmesh = mesh
    return newmesh


#
# def set_mesh_quadrature(mesh, cell, qdegree = None, vqdegree = None):
#
#     if qdegree:
#         if cell in ['interval', 'quad', 'tri', 'hex']:
#         	mesh.dx = dx(metadata={"quadrature_degree": qdegree})
#         	mesh.dS = dS(metadata={"quadrature_degree": qdegree})
#         	mesh.ds = ds(metadata={"quadrature_degree": qdegree})
# #EVENTUALLY THIS SHOULD ACTUALLY DO SEPARATE DEGREES IN HORIZ AND VERTICAL...
#         if cell in ['tpquad', 'tphex', 'tptri']:
#             vqdegree = vqdegree or qdegree
#         	mesh.dx = dx(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_b = ds_b(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_t = ds_t(metadata={"quadrature_degree": qdegree })
#         	mesh.ds_v = ds_v(metadata={"quadrature_degree": qdegree })
#         	mesh.dS_h = dS_h(metadata={"quadrature_degree": qdegree })
#         	mesh.dS_v = dS_v(metadata={"quadrature_degree": qdegree })
#         	mesh.dS = mesh.dS_h + mesh.dS_v
#         	mesh.ds_tb = mesh.ds_t + mesh.ds_b
#         	mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v
#     else:
#         if cell in ['interval', 'quad', 'tri', 'hex']:
#             mesh.dx = dx
#             mesh.dS = dS
#             mesh.ds = ds
#         if cell in ['tpquad', 'tphex', 'tptri']:
#             mesh.dx = dx
#             mesh.ds_b = ds_b
#             mesh.ds_v = ds_v
#             mesh.ds_t = ds_t
#             mesh.dS_h = dS_h
#             mesh.dS_v = dS_v
#             mesh.dS = mesh.dS_h + mesh.dS_v
#             mesh.ds_tb = mesh.ds_t + mesh.ds_b
#             mesh.ds = mesh.ds_b + mesh.ds_t + mesh.ds_v
#
# def set_boxmesh_operators(mesh, cell):
#
# # ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# # Basically what we need to make Hamiltonian and Hodge Laplacian stuff dimension-agnostic
# 	if cell in ['quad', 'tri', 'tpquad']:
# 		mesh.skewgrad = lambda s : perp(grad(s))
# 		mesh.perp = lambda u : perp(u)
#
#     if cell in ['tpquad']:
# 		mesh.k = as_vector((0,1))
#
#     if cell in ['hex', 'tphex', 'tptri']:
# 		#mesh.skewgrad = lambda s : cross(mesh.k,grad(s))
# 		#mesh.perp = lambda u : cross(mesh.k,u)
#         pass
#
# 	if cell in ['tphex','tptri']:
# 		mesh.k = as_vector((0,0,1))
#
# def make_sphere_normals(x):
# 	R = sqrt(inner(x, x))
# 	rhat_expr = x/R
# 	return rhat_expr
#
# # ADD OTHER OPERATORS HERE ie div, grad, curl, adjoints, etc.
# def set_spheremesh_operators(mesh, cell):
#
#     if cell in ['quad', 'tri']:
#     	xs = SpatialCoordinate(mesh)
#     	mesh.cell_normals = make_sphere_normals(xs)
#     	mesh.perp = lambda u: cross(mesh.cell_normals, u)
#     	mesh.skewgrad = lambda s: cross(mesh.cell_normals, grad(s))
#     	global_normal = as_vector((xs[0],xs[1],xs[2]))
#     	mesh.init_cell_orientations(global_normal)
# # FIX
# 	if cell in ['tphex','tptri']:
#         pass
