import jinja2
import numpy as np

from themis.themiselement import IntervalElement
from finat.point_set import TensorPointSet
from themis.constant import Constant
import themis.function as function

__all__ = ["generate_assembly_routine", "generate_interpolation_routine"]

exterior_facet_types = ['exterior_facet', 'exterior_facet_vert', 'exterior_facet_bottom', 'exterior_facet_top']
interior_facet_types = ['interior_facet', 'interior_facet_vert', 'interior_facet_horiz']

template_path = "/home/celdred/Dropbox/Research/Code/postdoc/firedrake-themis/gitrepo/themis"

# load templates and create environment
templateLoader = jinja2.FileSystemLoader(searchpath=template_path)
templateEnv = jinja2.Environment(loader=templateLoader, trim_blocks=True)

# read the templates
interpolation_template = templateEnv.get_template('interpolation.template')
cellassembly_template = templateEnv.get_template('assemble.template')
facetassembly_template = templateEnv.get_template('assemble-facets.template')


def a_to_cinit_string(x):
    np.set_printoptions(threshold=np.prod(x.shape))
    sx = np.array2string(x, separator=',', precision=100)
    sx = sx.replace('\n', '')
    sx = sx.replace('[', '{')
    sx = sx.replace(']', '}')
    return sx


class EntriesObject():
    def __init__(self, elem):

        subelemx = elem.sub_elements[0][0]
        self.entries_offset_x = subelemx.ofe
        self.entries_offset_mult_x = subelemx.ome
        self.nentries_x = subelemx.nentries

        subelemy = elem.sub_elements[0][1]
        self.entries_offset_y = subelemy.ofe
        self.entries_offset_mult_y = subelemy.ome
        self.nentries_y = subelemy.nentries

        subelemz = elem.sub_elements[0][2]
        self.entries_offset_z = subelemz.ofe
        self.entries_offset_mult_z = subelemz.ome
        self.nentries_z = subelemz.nentries

        self.nentries_total = elem.nentries[0]

        self.contx = subelemx.cont
        self.conty = subelemy.cont
        self.contz = subelemz.cont

        self.ndofs = elem.ndofs


class SpaceObject():
    def __init__(self, elem, name):
        self.name = name

        self.ncomp = elem.ncomp
        self.ndofs = elem.ndofs

        self.offsets_x = []
        self.offsets_y = []
        self.offsets_z = []
        self.offset_mult_x = []
        self.offset_mult_y = []
        self.offset_mult_z = []
        self.nbasis_x = []
        self.nbasis_y = []
        self.nbasis_z = []
        self.nblocks_x = []
        self.nblocks_y = []
        self.nblocks_z = []

        self.dalist = elem.dalist(name)
        self.fieldargs = elem.fieldargslist(name)

        for ci in range(self.ncomp):

            subelemx = elem.sub_elements[ci][0]
            self.offsets_x.append(subelemx.of)
            self.offset_mult_x.append(subelemx.om)
            self.nblocks_x.append(subelemx.nblocks)
            self.nbasis_x.append(subelemx.nbasis)

            subelemy = elem.sub_elements[ci][1]
            self.offsets_y.append(subelemy.of)
            self.offset_mult_y.append(subelemy.om)
            self.nblocks_y.append(subelemy.nblocks)
            self.nbasis_y.append(subelemy.nbasis)

            subelemz = elem.sub_elements[ci][2]
            self.offsets_z.append(subelemz.of)
            self.offset_mult_z.append(subelemz.om)
            self.nblocks_z.append(subelemz.nblocks)
            self.nbasis_z.append(subelemz.nbasis)

        self.nbasis = elem.nbasis
        self.nbasis_total = elem.nbasis_total


# Use tabdict to cache tab elems, which can be expensive to generate due to sympy lambdification of many basis functions
tabdict = {}


class TabObject():
    def __init__(self, tabulation, kernel, pts):

        self.name = tabulation['name']
        self.shape = tabulation['shape']

        if (tabulation['variant'], tabulation['cont'], tabulation['order']) in tabdict:
            tabelem = tabdict[(tabulation['variant'], tabulation['cont'], tabulation['order'])]
        else:
            tabelem = IntervalElement(tabulation['variant'], tabulation['cont'], tabulation['order'])
            tabdict[(tabulation['variant'], tabulation['cont'], tabulation['order'])] = tabelem

        self.vals = tabelem.tabulate_numerical(pts, tabulation['derivorder'])
        self.values = a_to_cinit_string(self.vals)
        self.npts = self.vals.shape[1]
        self.nbasis = self.vals.shape[2]
        self.nblocks = self.vals.shape[0]
        self.shiftaxis = tabulation['shiftaxis']

        self.cell = 'bad'
        # Key: we define 2 = - and 1 = +
        if kernel.integral_type in interior_facet_types:
            if tabulation['restrict'] == 'p':
                self.cell = '1'
            if tabulation['restrict'] == 'm':
                self.cell = '2'
        if kernel.integral_type in exterior_facet_types:
            if kernel.facet_exterior_boundary == 'lower':
                self.cell = '2'
            if kernel.facet_exterior_boundary == 'upper':
                self.cell = '1'


def get_pts(kernel, ndims, restrict, shiftaxis, shape):

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

    elif kernel.integral_type in exterior_facet_types + interior_facet_types:
        if restrict == 'm':
            zerodim_pts = np.zeros(shape[0])
        elif restrict == 'p':
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

    return allpts[shiftaxis]


def generate_assembly_routine(mesh, space1, space2, kernel):

    # determine space1 and space2 info
    if not (space1 is None):
        elem1 = space1.themis_element()
        space1obj = SpaceObject(elem1, 's1')

    if not (space2 is None):
        elem2 = space2.themis_element()
        space2obj = SpaceObject(elem2, 's2')

    if kernel.formdim == 2:
        matlist = ''
        for ci1 in range(space1obj.ncomp):
            for ci2 in range(space2obj.ncomp):
                matlist = matlist + ',' + 'Mat formmat_' + str(ci1) + '_' + str(ci2)
    if kernel.formdim == 1:
        veclist = ''
        for ci1 in range(space1obj.ncomp):
            veclist = veclist + ',' + 'Vec formvec_' + str(ci1)

    # get the list of fields and constants
    field_args_string = ''  # this goes in function call
    constant_args_string = ''  # this goes in function call
    kernel_args_string = ''  # this goes in kernel call
    fieldobjs = []

    if not kernel.zero:
        # coordinates
        fspace = mesh.coordinates.function_space().get_space(0)
        elem = fspace.get_space(0).themis_element()
        fieldobj = SpaceObject(elem, 'coords_0')
        field_args_string = field_args_string + fieldobj.fieldargs
        fieldobjs.append(fieldobj)
        kernel_args_string = kernel_args_string + ',' + '&' + 'coords_0_vals'

        # other fields
        for fieldindex in kernel.coefficient_map:
            field = kernel.coefficients[fieldindex]
            if isinstance(field, function.Function):
                for si in range(field.function_space().nspaces):
                    fspace = field.function_space().get_space(si)
                    elem = fspace.get_space(si).themis_element()
                    fieldobj = SpaceObject(elem, field.name() + '_' + str(si))
                    field_args_string = field_args_string + fieldobj.fieldargs
                    fieldobjs.append(fieldobj)
                    kernel_args_string = kernel_args_string + ',' + '&' + field.name() + '_' + str(si) + '_vals'
            if isinstance(field, Constant):
                constant_args_string = constant_args_string + ',' + 'const double *' + field.name()
                kernel_args_string = kernel_args_string + ',' + field.name()

    # do tabulations
    tabulations = []
    for tabulation in kernel.tabulations:
        pts = get_pts(kernel, mesh.ndim, tabulation['restrict'], tabulation['shiftaxis'], tabulation['shape'])
        tabobj = TabObject(tabulation, kernel, pts)
        tabulations.append(tabobj)

    # set template vars
    templateVars = {}

    if kernel.integral_type in interior_facet_types:
        templateVars['facet_type'] = 'interior'
        templateVars['facet_direc'] = kernel.facet_direc

    if kernel.integral_type in exterior_facet_types:
        templateVars['facet_type'] = 'exterior'
        templateVars['facet_exterior_boundary'] = kernel.facet_exterior_boundary
        templateVars['facet_direc'] = kernel.facet_direc

    templateVars['bcs'] = mesh.bcs

    if kernel.integral_type in ['interior_facet_horiz', 'exterior_facet_bottom', 'exterior_facet_top']:
        templateVars['extruded'] = 1
    else:
        templateVars['extruded'] = 0

    templateVars['tabulations'] = tabulations

    if kernel.zero:
        templateVars['kernelstr'] = ''
    else:
        templateVars['kernelstr'] = kernel.ast
        templateVars['kernelname'] = kernel.name

    templateVars['formdim'] = kernel.formdim
    templateVars['assemblytype'] = kernel.integral_type

    templateVars['ndim'] = mesh.ndim

    if kernel.formdim == 2:
        templateVars['submatlist'] = matlist
    if kernel.formdim == 1:
        templateVars['subveclist'] = veclist

    if not (space1 is None):
        templateVars['space1'] = space1obj

    if not (space2 is None):
        templateVars['space2'] = space2obj

    templateVars['fieldlist'] = fieldobjs
    templateVars['fieldargs'] = field_args_string
    templateVars['constantargs'] = constant_args_string
    templateVars['kernelargs'] = kernel_args_string

    # Process template to produce source code
    if kernel.integral_type in interior_facet_types + exterior_facet_types:
        outputText = facetassembly_template.render(templateVars)
    else:
        outputText = cellassembly_template.render(templateVars)

    return outputText


def generate_interpolation_routine(mesh, kernel):

    # Load element specific information- entries offsets/offset mult
    elem = kernel.elem
    entries = EntriesObject(elem)

    # get the list of fields and constants
    field_args_string = ''  # this goes in function call
    constant_args_string = ''  # this goes in function call
    kernel_args_string = ''  # this goes in kernel call
    fieldobjs = []
    for fieldindex in kernel.coefficient_map:
        field = kernel.coefficients[fieldindex]
        if isinstance(field, function.Function):
            for si in range(field.function_space().nspaces):
                fspace = field.function_space().get_space(si)
                elem = fspace.get_space(si).themis_element()
                fieldobj = SpaceObject(elem, field.name() + '_' + str(si))
                field_args_string = field_args_string + fieldobj.fieldargs
                fieldobjs.append(fieldobj)
                kernel_args_string = kernel_args_string + ',' + '&' + field.name() + '_' + str(si) + '_vals'
        if isinstance(field, Constant):
            constant_args_string = constant_args_string + ',' + 'const double *' + field.name()
            kernel_args_string = kernel_args_string + ',' + field.name()

    # do tabulations
    arrpts = np.array(kernel.pts)
    tabulations = []
    for tabulation in kernel.tabulations:
        pts = arrpts[:, tabulation['shiftaxis']]
        tabobj = TabObject(tabulation, kernel, pts)
        tabulations.append(tabobj)

    # Specify the input variables for the template
    templateVars = {}

    templateVars['tabulations'] = tabulations
    templateVars['bcs'] = mesh.bcs

    templateVars['kernelstr'] = kernel.ast
    templateVars['kernelname'] = kernel.name

    templateVars['ndim'] = mesh.ndim

    templateVars['entries'] = entries

    templateVars['fieldlist'] = fieldobjs
    templateVars['fieldargs'] = field_args_string
    templateVars['constantargs'] = constant_args_string
    templateVars['kernelargs'] = kernel_args_string

    # Process template to produce source code
    outputText = interpolation_template.render(templateVars)

    return outputText
