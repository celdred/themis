import jinja2
import numpy as np

from ufl import FiniteElement, interval
from finiteelement import ThemisElement
from finat.point_set import TensorPointSet
from function import Function, SplitFunction
from constant import Constant


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


exterior_facet_types = ['exterior_facet', 'exterior_facet_vert', 'exterior_facet_bottom', 'exterior_facet_top']
interior_facet_types = ['interior_facet', 'interior_facet_vert', 'interior_facet_horiz']

template_path = "/home/celdred/Dropbox/Research/Code/postdoc/firedrake-themis/gitrepo/themis"


def generate_assembly_routine(mesh, space1, space2, kernel):
    # load templates
    templateLoader = jinja2.FileSystemLoader(searchpath=template_path)

    # create environment
    templateEnv = jinja2.Environment(loader=templateLoader, trim_blocks=True)
    templateVars = {}

    ndims = mesh.ndim

    # read the template
    if kernel.integral_type in interior_facet_types + exterior_facet_types:
        template = templateEnv.get_template('assemble-facets.template')
    else:
        template = templateEnv.get_template('assemble.template')

    if kernel.integral_type in interior_facet_types:
        templateVars['facet_type'] = 'interior'
        templateVars['facet_direc'] = kernel.facet_direc

    if kernel.integral_type in exterior_facet_types:
        templateVars['facet_type'] = 'exterior'
        templateVars['facet_exterior_boundary'] = kernel.facet_exterior_boundary
        templateVars['facet_direc'] = kernel.facet_direc

    templateVars['bcs'] = mesh.bcs

    # Load element specific information- offsets/offset mult

    # THIS LOGIC CAN MAYBE MOVE TO FE?
    # IE AN ELEMENT INFO-TYPE THING?

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
        nblocks1_x = []
        nblocks1_y = []
        nblocks1_z = []
        s1dalist = ''

        elem1 = space1.themis_element()
        for ci1 in range(space1size):
            s1dalist = s1dalist + ',' + 'DM s1da_' + str(ci1)

            nb = elem1.get_nblocks(ci1, 0)
            of, ofm = elem1.get_offsets(ci1, 0)
            offsets1_x.append(a_to_cinit_string(of))
            offset_mult1_x.append(a_to_cinit_string(ofm))
            # offsets1_x.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult1_x.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks1_x.append(nb)
            nbasis1_x.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem1.get_nblocks(ci1, 1)
            of, ofm = elem1.get_offsets(ci1, 1)
            offsets1_y.append(a_to_cinit_string(of))
            offset_mult1_y.append(a_to_cinit_string(ofm))
            # offsets1_y.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult1_y.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks1_y.append(nb)
            nbasis1_y.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem1.get_nblocks(ci1, 2)
            of, ofm = elem1.get_offsets(ci1, 2)
            offsets1_z.append(a_to_cinit_string(of))
            offset_mult1_z.append(a_to_cinit_string(ofm))
            # offsets1_z.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult1_z.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks1_z.append(nb)
            nbasis1_z.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

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
        nblocks2_x = []
        nblocks2_y = []
        nblocks2_z = []
        s2dalist = ''

        elem2 = space2.themis_element()
        for ci2 in range(space2size):
            s2dalist = s2dalist + ',' + 'DM s2da_' + str(ci2)

            nb = elem2.get_nblocks(ci2, 0)
            of, ofm = elem2.get_offsets(ci2, 0)
            offsets2_x.append(a_to_cinit_string(of))
            offset_mult2_x.append(a_to_cinit_string(ofm))
            # offsets2_x.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult2_x.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks2_x.append(nb)
            nbasis2_x.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem2.get_nblocks(ci2, 1)
            of, ofm = elem2.get_offsets(ci2, 1)
            offsets2_y.append(a_to_cinit_string(of))
            offset_mult2_y.append(a_to_cinit_string(ofm))
            # offsets2_y.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult2_y.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks2_y.append(nb)
            nbasis2_y.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem2.get_nblocks(ci2, 2)
            of, ofm = elem2.get_offsets(ci2, 2)
            offsets2_z.append(a_to_cinit_string(of))
            offset_mult2_z.append(a_to_cinit_string(ofm))
            # offsets2_z.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # offset_mult2_z.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            nblocks2_z.append(nb)
            nbasis2_z.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

        nbasis2_total = np.sum(np.array(nbasis2_x, dtype=np.int32) * np.array(nbasis2_y, dtype=np.int32) * np.array(nbasis2_z, dtype=np.int32))

    # load fields info, including coordinates
    field_args_string = ''
    constant_args_string = ''
    fieldobjs = []
    # fielddict = {}
    fieldplusconstantslist = []

    # get the list of fields and constants
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
        fieldobj.nblocks_x = []
        fieldobj.nblocks_y = []
        fieldobj.nblocks_z = []
        fieldobj.ndofs = fspace.get_space(si).themis_element().ndofs()
        for ci in range(fspace.get_space(si).ncomp):
            elem = fspace.get_space(si).themis_element()

            nb = elem.get_nblocks(ci, 0)
            of, ofm = elem.get_offsets(ci, 0)
            fieldobj.offsets_x.append(a_to_cinit_string(of))
            fieldobj.offset_mult_x.append(a_to_cinit_string(ofm))
            # fieldobj.offsets_x.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # fieldobj.offset_mult_x.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            fieldobj.nblocks_x.append(nb)
            fieldobj.nbasis_x.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem.get_nblocks(ci, 1)
            of, ofm = elem.get_offsets(ci, 1)
            fieldobj.offsets_y.append(a_to_cinit_string(of))
            fieldobj.offset_mult_y.append(a_to_cinit_string(ofm))
            # fieldobj.offsets_y.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # fieldobj.offset_mult_y.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            fieldobj.nblocks_y.append(nb)
            fieldobj.nbasis_y.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem.get_nblocks(ci, 2)
            of, ofm = elem.get_offsets(ci, 2)
            fieldobj.offsets_z.append(a_to_cinit_string(of))
            fieldobj.offset_mult_z.append(a_to_cinit_string(ofm))
            # fieldobj.offsets_z.append(a_to_cinit_string(of[(nb-1)//2,:]))
            # fieldobj.offset_mult_z.append(a_to_cinit_string(ofm[(nb-1)//2,:]))
            fieldobj.nblocks_z.append(nb)
            fieldobj.nbasis_z.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            dmname = 'DM da_' + fieldobj.name + '_' + str(ci)
            vecname = 'Vec ' + fieldobj.name + '_' + str(ci)
            field_args_string = field_args_string + ', ' + dmname
            field_args_string = field_args_string + ', ' + vecname
        fieldobj.nbasis_total = np.sum(np.array(fieldobj.nbasis_x, dtype=np.int32) * np.array(fieldobj.nbasis_y, dtype=np.int32) * np.array(fieldobj.nbasis_z, dtype=np.int32))
        fieldobj.ncomp = fspace.get_space(si).ncomp

        fieldobjs.append(fieldobj)

    for constant in constantlist:
        constant_args_string = constant_args_string + ',' + 'double ' + constant.name()

    tabulations = []

    for tabulation in kernel.tabulations:

        tabobj = TabObject()
        tabobj.name = tabulation['name']

        # get the quadrature points for the tabulation
        shape = tabulation['shape']
        ps = kernel.finatquad.point_set

        if kernel.integral_type == 'cell':
            allpts = []
            if isinstance(ps, TensorPointSet):
                ps1, ps2 = ps.factors
                if isinstance(ps1, TensorPointSet):
                    subps1, subps2 = ps1.factors
                    allpts.append(subps1.points[:, 0])
                    allpts.append(subps2.points[:, 0])
                    allpts.append(ps2.points[:, 0])
                else:
                    allpts.append(ps1.points[:, 0])
                    allpts.append(ps2.points[:, 0])
            else:
                allpts.append(ps.points[:, 0])
            pts = allpts[tabulation['shiftaxis']]

        elif kernel.integral_type in exterior_facet_types + interior_facet_types:
            restrict = tabulation['restrict']

            if restrict == 'p':
                zerodim_pts = np.zeros(shape[0])
            elif restrict == 'm':
                zerodim_pts = np.ones(shape[0])
            elif restrict == '':
                if kernel.facet_exterior_boundary == 'upper':
                    zerodim_pts = np.ones(shape[0])
                elif kernel.facet_exterior_boundary == 'lower':
                    zerodim_pts = np.zeros(shape[0])

            if ndims == 1:  # pointwise integrals
                allpts = [zerodim_pts, None, None]

            elif ndims == 2:  # line integrals
                if kernel.integral_type in ['interior_facet_horiz', 'exterior_facet_top', 'exterior_facet_bottom']:  # always y facets
                    actualpts = ps.factors[0].points[:, 0]
                elif kernel.integral_type in ['interior_facet_vert', 'exterior_facet_vert']:  # always x facets
                    actualpts = ps.factors[1].points[:, 0]
                elif kernel.integral_type in ['interior_facet', 'exterior_facet']:  # x or y facets
                    actualpts = ps.points[:, 0]
                if kernel.facet_direc == 0:  # x facets
                    allpts = [zerodim_pts, actualpts, None]
                if kernel.facet_direc == 1:  # y facets
                    allpts = [actualpts, zerodim_pts, None]

            elif ndims == 3:  # facet integrals
                if kernel.integral_type in ['interior_facet_horiz', 'exterior_facet_top', 'exterior_facet_bottom']:  # always z facets
                    ptsx = ps.factors[0].factors[0].points[:, 0]
                    ptsy = ps.factors[0].factors[1].points[:, 0]
                    ptsz = zerodim_pts
                elif kernel.integral_type in ['interior_facet_vert', 'exterior_facet_vert']:
                    if kernel.facet_direc == 0:  # x facets
                        ptsy = ps.factors[0].points[:, 0]
                        ptsz = ps.factors[1].points[:, 0]
                        ptsx = zerodim_pts
                    if kernel.facet_direc == 1:  # z facets
                        ptsx = ps.factors[0].points[:, 0]
                        ptsz = ps.factors[1].points[:, 0]
                        ptsy = zerodim_pts
                elif kernel.integral_type in ['interior_facet', 'exterior_facet']:  # x, y or z facets
                    raise NotImplementedError('Facet integrals for non-TP in 3D are not yet implemented')

                allpts = [ptsx, ptsy, ptsz]

            pts = allpts[tabulation['shiftaxis']]

        if tabulation['discont'] is False:
            uflelem = FiniteElement("CG", interval, tabulation['order'], variant=tabulation['variant'])
        if tabulation['discont'] is True:
            uflelem = FiniteElement("DG", interval, tabulation['order'], variant=tabulation['variant'])
        tabelem = ThemisElement(uflelem)

        if tabulation['derivorder'] == 0:
            vals = tabelem.get_basis(0, 0, pts)
        if tabulation['derivorder'] == 1:
            vals = tabelem.get_derivs(0, 0, pts)
        if tabulation['derivorder'] == 2:
            vals = tabelem.get_derivs2(0, 0, pts)
# HACK
        # tabobj.values = a_to_cinit_string(vals[(vals.shape[0]-1)//2,:,:])
        tabobj.values = a_to_cinit_string(vals)
        tabobj.npts = vals.shape[1]
        tabobj.nbasis = vals.shape[2]
        tabobj.nblocks = vals.shape[0]
        tabobj.shiftaxis = tabulation['shiftaxis']

        tabobj.cell = 'bad'
        # Key: we define 1 = - and 2 = +
        if kernel.integral_type in interior_facet_types:
            if tabulation['restrict'] == 'p':
                tabobj.cell = '2'
            if tabulation['restrict'] == 'm':
                tabobj.cell = '1'
        if kernel.integral_type in exterior_facet_types:
            if kernel.facet_exterior_boundary == 'lower':
                tabobj.cell = '2'
            if kernel.facet_exterior_boundary == 'upper':
                tabobj.cell = '1'

        # sanity checks
        assert(tabulation['shape'] == vals.shape[1:])
        assert(tabelem.get_nblocks(0, 0) == vals.shape[0])

        tabulations.append(tabobj)

        # construct element (or get it from the list of fields/space1/space2/coords)?
        # add tabulations to templateVars
    templateVars['tabulations'] = tabulations

    if kernel.integral_type in ['interior_facet_horiz', 'exterior_facet_bottom', 'exterior_facet_top']:
        templateVars['extruded'] = 1
    else:
        templateVars['extruded'] = 0

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
        templateVars['nblocks1_x'] = nblocks1_x

        templateVars['offsets1_y'] = offsets1_y
        templateVars['offset_mult1_y'] = offset_mult1_y
        templateVars['nbasis1_y'] = nbasis1_y
        templateVars['nblocks1_y'] = nblocks1_y

        templateVars['offsets1_z'] = offsets1_z
        templateVars['offset_mult1_z'] = offset_mult1_z
        templateVars['nbasis1_z'] = nbasis1_z
        templateVars['nblocks1_z'] = nblocks1_z

    if not (space2 is None):
        templateVars['nci2'] = space2size
        templateVars['s2dalist'] = s2dalist
        templateVars['nbasis2_total'] = nbasis2_total

        templateVars['offsets2_x'] = offsets2_x
        templateVars['offset_mult2_x'] = offset_mult2_x
        templateVars['nbasis2_x'] = nbasis2_x
        templateVars['nblocks2_x'] = nblocks2_x

        templateVars['offsets2_y'] = offsets2_y
        templateVars['offset_mult2_y'] = offset_mult2_y
        templateVars['nbasis2_y'] = nbasis2_y
        templateVars['nblocks2_y'] = nblocks2_y

        templateVars['offsets2_z'] = offsets2_z
        templateVars['offset_mult2_z'] = offset_mult2_z
        templateVars['nbasis2_z'] = nbasis2_z
        templateVars['nblocks2_z'] = nblocks2_z

    # fields
    templateVars['fieldlist'] = fieldobjs
    templateVars['fieldargs'] = field_args_string

    # constants
    templateVars['fieldplusconstantslist'] = fieldplusconstantslist
    templateVars['constantargs'] = constant_args_string

    # Process template to produce source code
    outputText = template.render(templateVars)

    return outputText


def generate_evaluate_routine(mesh, kernel):
    # load templates
    templateLoader = jinja2.FileSystemLoader(searchpath=template_path)

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

    field = kernel.field
# EVALUATION ON AN UNSPLIT MIXED FIELD SHOULD FAIL!
    if isinstance(field, Function):
        if field.name() == mesh.coordinates.name():
            evalfieldindex = 0
        else:
            for si in range(field.function_space().nspaces):
                fieldlist.append((field, si))
            evalfieldindex = 1
    if isinstance(field, SplitFunction):
        for si in range(field._parentfunction.function_space().nspaces):
            fieldlist.append((field._parentfunction, si))
        evalfieldindex = field._si + 1

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
        fieldobj.nblocks_x = []
        fieldobj.nblocks_y = []
        fieldobj.nblocks_z = []
        fieldobj.ndofs = fspace.get_space(si).themis_element().ndofs()
        for ci in range(fspace.get_space(si).ncomp):
            elem = fspace.get_space(si).themis_element()

            nb = elem.get_nblocks(ci, 0)
            of, ofm = elem.get_offsets(ci, 0)
            b = elem.get_basis(ci, 0, pts[0])
            d = elem.get_derivs(ci, 0, pts[0])
            fieldobj.basis_x.append(a_to_cinit_string(b))
            fieldobj.derivs_x.append(a_to_cinit_string(d))
            fieldobj.offsets_x.append(a_to_cinit_string(of))
            fieldobj.offset_mult_x.append(a_to_cinit_string(ofm))
            fieldobj.nblocks_x.append(nb)
            fieldobj.nbasis_x.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem.get_nblocks(ci, 1)
            of, ofm = elem.get_offsets(ci, 1)
            b = elem.get_basis(ci, 1, pts[1])
            d = elem.get_derivs(ci, 1, pts[1])
            fieldobj.basis_y.append(a_to_cinit_string(b))
            fieldobj.derivs_y.append(a_to_cinit_string(d))
            fieldobj.offsets_y.append(a_to_cinit_string(of))
            fieldobj.offset_mult_y.append(a_to_cinit_string(ofm))
            fieldobj.nblocks_y.append(nb)
            fieldobj.nbasis_y.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            nb = elem.get_nblocks(ci, 2)
            of, ofm = elem.get_offsets(ci, 2)
            b = elem.get_basis(ci, 2, pts[2])
            d = elem.get_derivs(ci, 2, pts[2])
            fieldobj.basis_z.append(a_to_cinit_string(b))
            fieldobj.derivs_z.append(a_to_cinit_string(d))
            fieldobj.offsets_z.append(a_to_cinit_string(of))
            fieldobj.offset_mult_z.append(a_to_cinit_string(ofm))
            fieldobj.nblocks_z.append(nb)
            fieldobj.nbasis_z.append(of.shape[1])  # assumption that each block has the same number of basis functions, which is fundamental

            dmname = 'DM da_' + fieldobj.name + '_' + str(ci)
            vecname = 'Vec ' + fieldobj.name + '_' + str(ci)
            field_args_string = field_args_string + ', ' + dmname
            field_args_string = field_args_string + ', ' + vecname
        fieldobj.nbasis_total = np.sum(np.array(fieldobj.nbasis_x, dtype=np.int32) * np.array(fieldobj.nbasis_y, dtype=np.int32) * np.array(fieldobj.nbasis_z, dtype=np.int32))
        fieldobj.ncomp = fspace.get_space(si).ncomp
        fieldobjs.append(fieldobj)

    vals_args_string = ''
    dmname = 'DM da_vals'
    vecname = 'Vec evals'
    vals_args_string = vals_args_string + ', ' + dmname
    vals_args_string = vals_args_string + ', ' + vecname

    # Specify the input variables for the template

    templateVars['ndim'] = ndims
    templateVars['bcs'] = mesh.bcs

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
