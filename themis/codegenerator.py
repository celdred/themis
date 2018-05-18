import jinja2
import numpy as np
# from ufl import VectorElement
from function import Function
from constant import Constant
from quadrature import ThemisQuadratureFinat


def a_to_cinit_string(x):
    np.set_printoptions(threshold=np.prod(x.shape))
    sx = np.array2string(x, separator=',', precision=100)
    sx = sx.replace('\n', '')
    sx = sx.replace('[', '{')
    sx = sx.replace(']', '}')
    return sx

# Needed for code generation, just holds field specific stuff like offsets, etc.


class FieldObject():
    def __init__(self):
        pass


class TabObject():
    def __init__(self):
        pass


def generate_assembly_routine(mesh, space1, space2, kernel):
    # load templates
    templateLoader = jinja2.FileSystemLoader(searchpath=["../../gitrepo/themis","../gitrepo/themis" ])

    # create environment
    templateEnv = jinja2.Environment(loader=templateLoader, trim_blocks=True)
    templateVars = {}

    ndims = mesh.ndim

    # read the template
    if kernel.integral_type in ['interior_facet', 'exterior_facet']:
        template = templateEnv.get_template('assemble-facets.template')
    else:
        template = templateEnv.get_template('assemble.template')

    if kernel.integral_type == 'interior_facet':
        templateVars['facet_type'] = 'interior'
        templateVars['facet_direc'] = kernel.facet_direc

    if kernel.integral_type == 'exterior_facet':
        templateVars['facet_type'] = 'exterior'
        templateVars['facet_exterior_boundary'] = kernel.facet_exterior_boundary
        templateVars['facet_direc'] = kernel.facet_direc

    if kernel.integral_type in ['interior_facet', 'exterior_facet']:
        templateVars['bcs'] = mesh.bcs

    # Load element specific information- offsets/offset mult
    # bindices = [0, 0, 0]  # these are block indices to indicate which
    # For now, we only support a single type of elements
    # For MGD with boundaries, need to support multiple element types

    # get the offsets

    if not (space1 is None):
        space1size = space1.ncomp
        offsets1_x = []
        offsets1_y = []
        offsets1_z = []
        offset_mult1_x = []
        offset_mult1_y = []
        offset_mult1_z = []
        nbasis1_x = []
        nbasis1_y = []
        nbasis1_z = []
        s1dalist = ''

        elem1 = space1.themis_element()
        for ci1 in range(space1size):
            s1dalist = s1dalist + ',' + 'DM s1da_' + str(ci1)

            # THESE SHOULD EVENTUALLY TAKE A BLOCK INDEX
            ofx, ofmx = elem1.get_offsets(ci1, 0)
            ofy, ofmy = elem1.get_offsets(ci1, 1)
            ofz, ofmz = elem1.get_offsets(ci1, 2)
            offsets1_x.append(a_to_cinit_string(ofx))
            offsets1_y.append(a_to_cinit_string(ofy))
            offsets1_z.append(a_to_cinit_string(ofz))
            offset_mult1_x.append(a_to_cinit_string(ofmx))
            offset_mult1_y.append(a_to_cinit_string(ofmy))
            offset_mult1_z.append(a_to_cinit_string(ofmz))
            nbasis1_x.append(len(ofx))
            nbasis1_y.append(len(ofy))
            nbasis1_z.append(len(ofz))
        nbasis1_total = np.sum(np.array(nbasis1_x, dtype=np.int32) * np.array(nbasis1_y, dtype=np.int32) * np.array(nbasis1_z, dtype=np.int32))

    if not (space2 is None):
        space2size = space2.ncomp
        offsets2_x = []
        offsets2_y = []
        offsets2_z = []
        offset_mult2_x = []
        offset_mult2_y = []
        offset_mult2_z = []
        nbasis2_x = []
        nbasis2_y = []
        nbasis2_z = []
        s2dalist = ''

        elem2 = space2.themis_element()
        for ci2 in range(space2size):
            s2dalist = s2dalist + ',' + 'DM s2da_' + str(ci2)

            # THESE SHOULD EVENTUALLY TAKE A BLOCK INDEX
            ofx, ofmx = elem2.get_offsets(ci2, 0)
            ofy, ofmy = elem2.get_offsets(ci2, 1)
            ofz, ofmz = elem2.get_offsets(ci2, 2)
            offsets2_x.append(a_to_cinit_string(ofx))
            offsets2_y.append(a_to_cinit_string(ofy))
            offsets2_z.append(a_to_cinit_string(ofz))
            offset_mult2_x.append(a_to_cinit_string(ofmx))
            offset_mult2_y.append(a_to_cinit_string(ofmy))
            offset_mult2_z.append(a_to_cinit_string(ofmz))
            nbasis2_x.append(len(ofx))
            nbasis2_y.append(len(ofy))
            nbasis2_z.append(len(ofz))
        nbasis2_total = np.sum(np.array(nbasis2_x, dtype=np.int32) * np.array(nbasis2_y, dtype=np.int32) * np.array(nbasis2_z, dtype=np.int32))

    # load fields info, including coordinates
# REVISE
    field_args_string = ''
    constant_args_string = ''
    fieldobjs = []
    # fielddict = {}
    fieldplusconstantslist = []

    # get the list of fields and constants
# REVISE
    fieldlist = []
    constantlist = []
    if not kernel.zero:
        fieldlist.append((mesh.coordinates, 0))
        fieldplusconstantslist.append(mesh.coordinates.name() + '_' + str(0) + '_vals')
        for fieldindex in kernel.coefficient_map:
            field = kernel.coefficients[fieldindex]
            if isinstance(field, Function):
                for si in range(field.function_space().nspaces):
                    fieldlist.append((field, si))
                    fieldplusconstantslist.append(field.name() + '_' + str(si) + '_vals')
            if isinstance(field, Constant):
                constantlist.append(field)
                fieldplusconstantslist.append('&' + field.name())
                # fieldplusconstantslist.append(field.name())
                # BROKEN FOR VECTOR/TENSOR CONSTANTS

    # print fieldplusconstantslist
    # print kernel.ast
# REVISE
    for field, si in fieldlist:
        fspace = field.function_space().get_space(si)
        fieldobj = FieldObject()
        fieldobj.name = field.name() + '_' + str(si)
        fieldobj.nbasis_x = []
        fieldobj.nbasis_y = []
        fieldobj.nbasis_z = []
        fieldobj.offsets_x = []
        fieldobj.offsets_y = []
        fieldobj.offsets_z = []
        fieldobj.offset_mult_x = []
        fieldobj.offset_mult_y = []
        fieldobj.offset_mult_z = []

        fieldobj.ndofs = fspace.get_space(si).themis_element().ndofs()
        for ci in range(fspace.get_space(si).ncomp):
            elem = fspace.get_space(si).themis_element()

            # THESE SHOULD EVENTUALLY TAKE A BLOCK INDEX
            ofx, ofmx = elem.get_offsets(ci, 0)
            ofy, ofmy = elem.get_offsets(ci, 1)
            ofz, ofmz = elem.get_offsets(ci, 2)
            fieldobj.offsets_x.append(a_to_cinit_string(ofx))
            fieldobj.offsets_y.append(a_to_cinit_string(ofy))
            fieldobj.offsets_z.append(a_to_cinit_string(ofz))
            fieldobj.offset_mult_x.append(a_to_cinit_string(ofmx))
            fieldobj.offset_mult_y.append(a_to_cinit_string(ofmy))
            fieldobj.offset_mult_z.append(a_to_cinit_string(ofmz))
            fieldobj.nbasis_x.append(len(ofx))
            fieldobj.nbasis_y.append(len(ofy))
            fieldobj.nbasis_z.append(len(ofz))

            dmname = 'DM da_' + fieldobj.name + '_' + str(ci)
            vecname = 'Vec ' + fieldobj.name + '_' + str(ci)
            field_args_string = field_args_string + ', ' + dmname
            field_args_string = field_args_string + ', ' + vecname
        fieldobj.nbasis_total = np.sum(np.array(fieldobj.nbasis_x, dtype=np.int32) * np.array(fieldobj.nbasis_y, dtype=np.int32) * np.array(fieldobj.nbasis_z, dtype=np.int32))
        fieldobj.ncomp = fspace.get_space(si).ncomp

        fieldobjs.append(fieldobj)

# REVISE
    for constant in constantlist:
        constant_args_string = constant_args_string + ',' + 'double ' + constant.name()
        # BROKEN FOR VECTOR/TENSOR CONSTANTS

    # This is just a Coefficient, but we are putting data INTO it!
#    if kernel.evaluate:
 #       vals_args_string = ''
  #      dmname = 'DM da_vals'
  #      vecname = 'Vec evals'
   #     vals_args_string = vals_args_string + ', ' + dmname
   #     vals_args_string = vals_args_string + ', ' + vecname

    tabulations = []

    quad = ThemisQuadratureFinat(kernel.finatquad)
    pts = quad.get_pts()

    # This stuff is likely WRONG for 3D/Extruded, and possibly for 1D?
    # for 3D/extruded I think I just need to be careful for how y/z, etc. are obtained- ie quad.get_pts might return stuff in the wrong order!
    if kernel.integral_type in ['interior_facet', 'exterior_facet']:
        if kernel.facet_direc == 0:
            y, _, _ = quad.get_pts()
            one = np.ones(y.shape[0])
            zero = np.zeros(y.shape[0])
            pts_pos = [zero, y, None]
            pts_neg = [one, y, None]
        if kernel.facet_direc == 1:
            x, _, _ = quad.get_pts()
            one = np.ones(x.shape[0])
            zero = np.zeros(x.shape[0])
            pts_pos = [x, zero, None]
            pts_neg = [x, one, None]
    if kernel.facet_exterior_boundary == 'upper':
        pts = pts_neg
    if kernel.facet_exterior_boundary == 'lower':
        pts = pts_pos

    # print(kernel.integral_type, kernel.facet_direc,kernel.facet_exterior_boundary)
    from ufl import FiniteElement, interval
    from finiteelement import ThemisElement
    for tabulation in kernel.tabulations:

        restrict = tabulation['restriction']
        if restrict == 'p':
            pts = pts_pos
        if restrict == 'm':
            pts = pts_neg

        tabobj = TabObject()
        tabobj.name = tabulation['name']

        if tabulation['discont'] is False:
            uflelem = FiniteElement("CG", interval, tabulation['order'], variant=tabulation['variant'])
        if tabulation['discont'] is True:
            uflelem = FiniteElement("DG", interval, tabulation['order'], variant=tabulation['variant'])
        tabelem = ThemisElement(uflelem)
        if tabulation['derivorder'] == 0:
            vals = tabelem.get_basis(0, 0, pts[tabulation['shiftaxis']])
        if tabulation['derivorder'] == 1:
            vals = tabelem.get_derivs(0, 0, pts[tabulation['shiftaxis']])
        if tabulation['derivorder'] == 2:
            vals = tabelem.get_derivs2(0, 0, pts[tabulation['shiftaxis']])
        tabobj.values = a_to_cinit_string(vals)
        tabobj.npts = vals.shape[0]
        tabobj.nbasis = vals.shape[1]
        tabulations.append(tabobj)

        # print(tabulation)
        # print(uflelem)
        # print(tabulation['order'],tabobj.npts,tabobj.nbasis)
        # print(pts[tabulation['shiftaxis']].shape)
        # print(tabelem.get_basis(0,0,pts[tabulation['shiftaxis']]).shape)
        # print(tabelem.get_derivs(0,0,pts[tabulation['shiftaxis']]).shape)

        # construct element (or get it from the list of fields/space1/space2/coords)?
        # add tabulations to templateVars
    templateVars['tabulations'] = tabulations

    if kernel.formdim == 2:
        matlist = ''
        for ci1 in range(space1size):
            for ci2 in range(space2size):
                matlist = matlist + ',' + 'Mat formmat_' + str(ci1) + '_' + str(ci2)
    if kernel.formdim == 1:
        veclist = ''
        for ci1 in range(space1size):
            veclist = veclist + ',' + 'Vec formvec_' + str(ci1)

    # Specific the input variables for the template
    if kernel.zero:
        templateVars['kernelstr'] = ''
    else:
        templateVars['kernelstr'] = kernel.ast
        templateVars['kernelname'] = kernel.name

    templateVars['formdim'] = kernel.formdim
    templateVars['assemblytype'] = kernel.integral_type

    templateVars['ndim'] = ndims

    if kernel.formdim == 2:
        templateVars['submatlist'] = matlist
    if kernel.formdim == 1:
        templateVars['subveclist'] = veclist

    # basis/derivs/etc.
    if not (space1 is None):
        templateVars['nci1'] = space1size
        templateVars['s1dalist'] = s1dalist
        templateVars['nbasis1_total'] = nbasis1_total

        templateVars['offsets1_x'] = offsets1_x
        templateVars['offset_mult1_x'] = offset_mult1_x
        templateVars['nbasis1_x'] = nbasis1_x

        templateVars['offsets1_y'] = offsets1_y
        templateVars['offset_mult1_y'] = offset_mult1_y
        templateVars['nbasis1_y'] = nbasis1_y

        templateVars['offsets1_z'] = offsets1_z
        templateVars['offset_mult1_z'] = offset_mult1_z
        templateVars['nbasis1_z'] = nbasis1_z

    if not (space2 is None):
        templateVars['nci2'] = space2size
        templateVars['s2dalist'] = s2dalist
        templateVars['nbasis2_total'] = nbasis2_total

        templateVars['offsets2_x'] = offsets2_x
        templateVars['offset_mult2_x'] = offset_mult2_x
        templateVars['nbasis2_x'] = nbasis2_x

        templateVars['offsets2_y'] = offsets2_y
        templateVars['offset_mult2_y'] = offset_mult2_y
        templateVars['nbasis2_y'] = nbasis2_y

        templateVars['offsets2_z'] = offsets2_z
        templateVars['offset_mult2_z'] = offset_mult2_z
        templateVars['nbasis2_z'] = nbasis2_z

    # fields
# REVISE
    templateVars['fieldlist'] = fieldobjs
    templateVars['fieldargs'] = field_args_string

    # constants
# REVISE
    templateVars['fieldplusconstantslist'] = fieldplusconstantslist
    templateVars['constantargs'] = constant_args_string

#    if kernel.evaluate:
#        templateVars['evaluate'] = 1
#        templateVars['valsargs'] = vals_args_string
#    else:
    templateVars['evaluate'] = 0

    # FIX THIS- HOW DO WE DETERMINE MATRIX FREE?
    # HOW ARE MATRIX-FREE KERNELS SUPPORTED?

    # if functional.assemblytype == 'tensor-product-matrixfree':
    # templateVars['matrixfree'] = 1
    # else:
    # templateVars['matrixfree'] = 0

    # Process template to produce source code
    outputText = template.render(templateVars)

    return outputText


def generate_evaluate_routine(mesh, kernel):
    # load templates
    templateLoader = jinja2.FileSystemLoader(searchpath=["../../gitrepo/themis","../gitrepo/themis" ])

    # create environment
    templateEnv = jinja2.Environment(loader=templateLoader, trim_blocks=True)
    templateVars = {}

    ndims = mesh.ndim

    # read the template
    template = templateEnv.get_template('evaluate.template')

    # load fields info, including coordinates
    field_args_string = ''
    fieldobjs = []


    # get the list of fields
    fieldlist = []
    fieldlist.append((mesh.coordinates, 0))
    for fieldindex in kernel.coefficient_map:
        field = kernel.coefficients[fieldindex]
        if field.name() == mesh.coordinates.name():
            evalfieldindex = 0
            continue
        else:
            evalfieldindex = 1
        for si in range(field.function_space().nspaces):
            fieldlist.append((field, si))


    pts = kernel.quad.get_pts()
    
    for field, si in fieldlist:
        fspace = field.function_space().get_space(si)
        fieldobj = FieldObject()
        fieldobj.name = field.name() + '_' + str(si)
        fieldobj.nbasis_x = []
        fieldobj.nbasis_y = []
        fieldobj.nbasis_z = []
        fieldobj.offsets_x = []
        fieldobj.offsets_y = []
        fieldobj.offsets_z = []
        fieldobj.offset_mult_x = []
        fieldobj.offset_mult_y = []
        fieldobj.offset_mult_z = []
        fieldobj.basis_x = []
        fieldobj.basis_y = []
        fieldobj.basis_z = []
        fieldobj.derivs_x = []
        fieldobj.derivs_y = []
        fieldobj.derivs_z = []
        
        fieldobj.ndofs = fspace.get_space(si).themis_element().ndofs()
        for ci in range(fspace.get_space(si).ncomp):
            elem = fspace.get_space(si).themis_element()

            # THESE SHOULD EVENTUALLY TAKE A BLOCK INDEX
            ofx, ofmx = elem.get_offsets(ci, 0)
            ofy, ofmy = elem.get_offsets(ci, 1)
            ofz, ofmz = elem.get_offsets(ci, 2)
            bx = elem.get_basis(ci,0,pts[0])
            by = elem.get_basis(ci,1,pts[1])
            bz = elem.get_basis(ci,2,pts[2])
            dx = elem.get_derivs(ci,0,pts[0])
            dy = elem.get_derivs(ci,1,pts[1])
            dz = elem.get_derivs(ci,2,pts[2])
            fieldobj.offsets_x.append(a_to_cinit_string(ofx))
            fieldobj.offsets_y.append(a_to_cinit_string(ofy))
            fieldobj.offsets_z.append(a_to_cinit_string(ofz))
            fieldobj.offset_mult_x.append(a_to_cinit_string(ofmx))
            fieldobj.offset_mult_y.append(a_to_cinit_string(ofmy))
            fieldobj.offset_mult_z.append(a_to_cinit_string(ofmz))

            fieldobj.basis_x.append(a_to_cinit_string(bx))
            fieldobj.basis_y.append(a_to_cinit_string(by))
            fieldobj.basis_z.append(a_to_cinit_string(bz))
            fieldobj.derivs_x.append(a_to_cinit_string(dx))
            fieldobj.derivs_y.append(a_to_cinit_string(dy))
            fieldobj.derivs_z.append(a_to_cinit_string(dz))
            
            fieldobj.nbasis_x.append(len(ofx))
            fieldobj.nbasis_y.append(len(ofy))
            fieldobj.nbasis_z.append(len(ofz))

            dmname = 'DM da_' + fieldobj.name + '_' + str(ci)
            vecname = 'Vec ' + fieldobj.name + '_' + str(ci)
            field_args_string = field_args_string + ', ' + dmname
            field_args_string = field_args_string + ', ' + vecname
        fieldobj.nbasis_total = np.sum(np.array(fieldobj.nbasis_x, dtype=np.int32) * np.array(fieldobj.nbasis_y, dtype=np.int32) * np.array(fieldobj.nbasis_z, dtype=np.int32))
        fieldobj.ncomp = fspace.get_space(si).ncomp
        #print(fieldobj.name,fieldobj.basis_z[0],pts[2])
        fieldobjs.append(fieldobj)


    vals_args_string = ''
    dmname = 'DM da_vals'
    vecname = 'Vec evals'
    vals_args_string = vals_args_string + ', ' + dmname
    vals_args_string = vals_args_string + ', ' + vecname

    # Specify the input variables for the template

    templateVars['ndim'] = ndims


    # fields
    templateVars['fieldlist'] = fieldobjs
    templateVars['fieldargs'] = field_args_string

    templateVars['npts_x'] = kernel.quad.nquadpts[0]
    templateVars['npts_y'] = kernel.quad.nquadpts[1]
    templateVars['npts_z'] = kernel.quad.nquadpts[2]

    templateVars['valsargs'] = vals_args_string
    templateVars['valstype'] = kernel.ctype
    templateVars['evaltype'] = kernel.etype
    templateVars['evalfield'] = fieldobjs[evalfieldindex]

    # Process template to produce source code
    outputText = template.render(templateVars)

    return outputText
