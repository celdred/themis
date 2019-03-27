
from numpy.ctypeslib import ndpointer
from pyop2.utils import get_petsc_dir
from pyop2 import compilation
import ctypes
from themis.codegenerator import generate_assembly_routine, generate_interpolation_routine
import themis.function as function
import themis.constant as constant

__all__ = ["extract_fields", "compile_functional"]


class CompiledKernel(object):
    base_cppargs = ["-I%s/include" % d for d in get_petsc_dir()]
    base_ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + ["-lpetsc", "-lm"]

    def __init__(self, source, function_name, restype=ctypes.c_int, cppargs=None, argtypes=None, comm=None):
        if cppargs is None:
            cppargs = self.base_cppargs
        else:
            cppargs = cppargs + self.base_cppargs

        funptr = compilation.load(source, "c", function_name,
                                  cppargs=cppargs,
                                  ldargs=self.base_ldargs,
                                  restype=restype,
                                  argtypes=argtypes,
                                  comm=comm)

        self.funptr = funptr

    def __call__(self, petsc_args, constant_args):
        args = [p.handle for p in petsc_args] + constant_args
        return self.funptr(*args)


def compile_functional(kernel, tspace, sspace, mesh):

    # create da list
    kernel.dalist = []
    facet_direc_list = []
    facet_exterior_boundary_list = []

    if kernel.integral_type == 'cell':
        kernel.dalist.append(mesh.cell_da)
        facet_direc_list.append('')
        facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'interior_facet':
        kernel.dalist.append(mesh.cell_da)
        facet_direc_list.append(0)
        facet_exterior_boundary_list.append('')
        if mesh.ndim >= 2:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')
        if mesh.ndim >= 3:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'exterior_facet':
        if mesh.bcs[0] == 'nonperiodic':
            kernel.dalist.append(mesh.cell_da)
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('lower')
        if mesh.ndim >= 2 and mesh.bcs[1] == 'nonperiodic':
            kernel.dalist.append(mesh.cell_da)
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')
        if mesh.ndim >= 3 and mesh.bcs[2] == 'nonperiodic':
            kernel.dalist.append(mesh.cell_da)
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('lower')

    if kernel.integral_type == 'interior_facet_vert':  # side facets
        kernel.dalist.append(mesh.cell_da)
        facet_direc_list.append(0)
        facet_exterior_boundary_list.append('')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'interior_facet_horiz':  # extruded facets
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('')

    if kernel.integral_type == 'exterior_facet_vert':  # side exterior facets
        if mesh.bcs[0] == 'nonperiodic':
            kernel.dalist.append(mesh.cell_da)
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(0)
            facet_exterior_boundary_list.append('lower')
        if mesh.extrusion_dim == 2 and mesh.bcs[1] == 'nonperiodic':
            kernel.dalist.append(mesh.cell_da)
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')

    if kernel.integral_type == 'exterior_facet_top':  # extruded exterior facet top
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('upper')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('upper')

    if kernel.integral_type == 'exterior_facet_bottom':  # extruded exterior facet bottom
        if mesh.extrusion_dim == 1:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(1)
            facet_exterior_boundary_list.append('lower')
        if mesh.extrusion_dim == 2:
            kernel.dalist.append(mesh.cell_da)
            facet_direc_list.append(2)
            facet_exterior_boundary_list.append('lower')

    # create field args list: matches construction used in codegenerator.py

    kernel.fieldargs_list = []
    kernel.constantargs_list = []
    fieldargtypeslist = []
    constantargtypeslist = []

    # append coordinates
    if not kernel.zero:  # don't build any field args stuff for zero kernel
        if not kernel.interpolate:
            kernel.fieldargs_list.append(kernel.mesh.coordinates.function_space().get_da(0))
            kernel.fieldargs_list.append(kernel.mesh.coordinates._lvectors[0])
            fieldargtypeslist.append(ctypes.c_voidp)
            fieldargtypeslist.append(ctypes.c_voidp)

        for fieldindex in kernel.coefficient_map:
            field = kernel.coefficients[fieldindex]
            if isinstance(field, function.Function):
                for si in range(field.function_space().nspaces):
                    fspace = field.function_space().get_space(si)
                    for ci in range(fspace.ncomp):
                        kernel.fieldargs_list.append(fspace.get_da(ci))
                        kernel.fieldargs_list.append(field.get_lvec(si, ci))
                        fieldargtypeslist.append(ctypes.c_voidp)  # DA
                        fieldargtypeslist.append(ctypes.c_voidp)  # Vec
            if isinstance(field, constant.Constant):
                constantargtypeslist.append(ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"))

    kernel.assemblyfunc_list = []
    tensorlist = []

    if (tspace is not None) and (sspace is not None):  # 2-form
        for ci1 in range(tspace.ncomp):  # Mat
            for ci2 in range(sspace.ncomp):
                tensorlist.append(ctypes.c_voidp)
        for ci1 in range(tspace.ncomp):  # tda
            tensorlist.append(ctypes.c_voidp)
        for ci2 in range(sspace.ncomp):  # sda
            tensorlist.append(ctypes.c_voidp)

    if (tspace is not None) and (sspace is None):  # 1-form

        for ci1 in range(tspace.ncomp):  # Vec
            tensorlist.append(ctypes.c_voidp)
        for ci1 in range(tspace.ncomp):  # tda
            tensorlist.append(ctypes.c_voidp)

    if kernel.interpolate is True:
        tensorlist.append(ctypes.c_voidp)  # DM
        tensorlist.append(ctypes.c_voidp)  # Vec

    argtypeslist = [ctypes.c_voidp, ] + tensorlist + fieldargtypeslist + constantargtypeslist

    restype = ctypes.c_int
    if (tspace is None) and (sspace is None):
        restype = ctypes.c_double  # 0-form

    for facet_direc, facet_exterior_boundary in zip(facet_direc_list, facet_exterior_boundary_list):
        kernel.facet_direc = facet_direc
        kernel.facet_exterior_boundary = facet_exterior_boundary

        if kernel.interpolate is True:
            assembly_routine = generate_interpolation_routine(mesh, kernel)
        else:
            assembly_routine = generate_assembly_routine(mesh, tspace, sspace, kernel)

        assembly_function = CompiledKernel(assembly_routine, "assemble", cppargs=["-O3"], argtypes=argtypeslist, restype=restype)
        kernel.assemblyfunc_list.append(assembly_function)
        kernel.argtypeslist = argtypeslist
        kernel.assembly_routine = assembly_routine

    if not kernel.zero:
        kernel.assemblycompiled = True


def extract_fields(kernel):

    if not kernel.zero:

        kernel.constantargs_list = []

        kernel.mesh.coordinates.scatter()

        for fieldindex in kernel.coefficient_map:
            field = kernel.coefficients[fieldindex]

            if isinstance(field, function.Function):
                field.scatter()

            if isinstance(field, constant.Constant):
                kernel.constantargs_list.append(field.dat)
