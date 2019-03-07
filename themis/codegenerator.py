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


class EmptyObject():
    def __init__(self):
        pass


exterior_facet_types = ['exterior_facet', 'exterior_facet_vert', 'exterior_facet_bottom', 'exterior_facet_top']
interior_facet_types = ['interior_facet', 'interior_facet_vert', 'interior_facet_horiz']

template_path = "/home/celdred/Dropbox/Research/Code/postdoc/firedrake-themis/gitrepo/themis"

def create_spaceobj(space):
    
    elem = space.themis_element()
    nbasis_x,nbasis_y,nbasis_z,nbasis,nbasis_total = elem.get_nbasis_info()
    cellx,celly,cellz,location,dofnum = elem.get_offset_info()

    spaceobj = EmptyObject()
    spaceobj.nbasis_x = a_to_cinit_string(nbasis_x)
    spaceobj.nbasis_y = a_to_cinit_string(nbasis_y)
    spaceobj.nbasis_z = a_to_cinit_string(nbasis_z)
    spaceobj.nbasis = a_to_cinit_string(nbasis)
    spaceobj.nbasis_total = nbasis_total
    spaceobj.cellx = a_to_cinit_string(cellx)
    spaceobj.celly = a_to_cinit_string(celly)
    spaceobj.cellz = a_to_cinit_string(cellz)
    spaceobj.location = location
    spaceobj.dofnum = dofnum
    spaceobj.ncomp = elem.ncomp
    spaceobj.ndofs = elem.ndofs()
    
    # ADD BLOCKS STUFF
    # ADD BASIS FUNCTIONS (NEEDED FOR EVALUATE)

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

    if not (space1 is None):
        space1obj = create_spaceobj(space1)

    if not (space2 is None):
        space2obj = create_spaceobj(space2)

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
        
        fieldobj = create_spaceobj(field.function_space().get_space(si))
        fieldobj.name = field.name() + '_' + str(si)
     
        dmname = 'DM da_' + fieldobj.name 
        vecname = 'Vec ' + fieldobj.name
        field_args_string = field_args_string + ', ' + dmname
        field_args_string = field_args_string + ', ' + vecname

        fieldobjs.append(fieldobj)

    for constant in constantlist:
        constant_args_string = constant_args_string + ',' + 'double ' + constant.name()

    tabulations = []

    for tabulation in kernel.tabulations:

        tabobj = EmptyObject()
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

    # Specific the input variables for the template
    if kernel.zero:
        templateVars['kernelstr'] = ''
    else:
        templateVars['kernelstr'] = kernel.ast
        templateVars['kernelname'] = kernel.name

    templateVars['formdim'] = kernel.formdim
    templateVars['assemblytype'] = kernel.integral_type

    templateVars['ndim'] = ndims

    # basis/derivs/etc.
    if not (space1 is None):
        templateVars['space1'] = space1obj


    if not (space2 is None):
        templateVars['space2'] = space2obj

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

        fieldobj = create_spaceobj(field.function_space().get_space(si))
        fieldobj.name = field.name() + '_' + str(si)
        fieldobjs.append(fieldobj)
       
        dmname = 'DM da_' + fieldobj.name
        vecname = 'Vec ' + fieldobj.name
        field_args_string = field_args_string + ', ' + dmname
        field_args_string = field_args_string + ', ' + vecname
        

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
