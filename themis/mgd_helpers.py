
# from petscshim import PETSc
from ufl import FiniteElement, TensorProductElement, interval, quadrilateral, HDivElement, hexahedron, HCurlElement, EnrichedElement, MixedElement, VectorElement
from functionspace import FunctionSpace, MixedFunctionSpace
from ufl_expr import TestFunction, TrialFunction, TestFunctions, TrialFunctions
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags


class Replacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def expr(self, o, *args):
        if o in self.mapping:
            return self.mapping[o]
        else:
            return self.reuse_if_untouched(o, *args)


def lower_element_order(elem):
    if isinstance(elem, VectorElement):
        ndofs = elem.value_size()
        baseelem = lower_element_order(elem.sub_elements()[0])
        return VectorElement(baseelem, dim=ndofs)
    elif isinstance(elem, MixedElement):  # must come after because VectorElement is a subclass of MixedElement!
        elemlist = elem.sub_elements()
        newelemlist = []
        for subelem in elemlist:
            newelemlist.append(lower_element_order(subelem))
        return newelemlist

    if isinstance(elem, EnrichedElement):
        elem1, elem2 = elem._elements
        if (isinstance(elem1, HDivElement) and isinstance(elem2, HDivElement)):
            elem1 = elem1._element
            elem2 = elem2._element
            elem1low = lower_element_order(elem1)
            elem2low = lower_element_order(elem2)
            return HDivElement(elem1low) + HDivElement(elem2low)
        if (isinstance(elem1, HCurlElement) and isinstance(elem2, HCurlElement)):
            elem1 = elem1._element
            elem2 = elem2._element
            elem1low = lower_element_order(elem1)
            elem2low = lower_element_order(elem2)
            return HCurlElement(elem1low) + HCurlElement(elem2low)
    elif isinstance(elem, TensorProductElement):
        elem1, elem2 = elem.sub_elements()
        elem1low = lower_element_order(elem1)
        elem2low = lower_element_order(elem2)
        return TensorProductElement(elem1low, elem2low)
    elif isinstance(elem, FiniteElement):
        if elem.cell().cellname() == 'interval':
            if elem.family() == 'Discontinuous Lagrange':
                return FiniteElement("DG", interval, 0, variant='mgd')
            elif elem.family() == 'Lagrange':
                return FiniteElement("CG", interval, 1, variant='mgd')
        elif elem.cell().cellname() == 'quadrilateral':
            if elem.family() == 'Q':
                return FiniteElement("Q", quadrilateral, 1, variant='mgd')
            if elem.family() == 'DQ':
                return FiniteElement("DQ", quadrilateral, 0, variant='mgd')
            if elem.family() == 'RTCF':
                return FiniteElement("RTCF", quadrilateral, 1, variant='mgd')
            if elem.family() == 'RTCE':
                return FiniteElement("RTCE", quadrilateral, 1, variant='mgd')
        elif elem.cell().cellname() == 'hexahedron':
            if elem.family() == 'Q':
                return FiniteElement("Q", hexahedron, 1, variant='mgd')
            if elem.family() == 'DQ':
                return FiniteElement("DQ", hexahedron, 0, variant='mgd')
            if elem.family() == 'NCF':
                return FiniteElement("NCF", hexahedron, 1, variant='mgd')
            if elem.family() == 'NCE':
                return FiniteElement("NCE", hexahedron, 1, variant='mgd')


def lower_form_order(form):
    # get test and trial functions of (possibly mixed) form
    testfunc, trialfunc = form.arguments()
    mesh = testfunc.ufl_domain()

    testelem = testfunc.ufl_element()
    trialelem = trialfunc.ufl_element()
    assert(testelem == trialelem)

    elem_lowest = lower_element_order(testelem)
    ismixed = False
    if isinstance(testelem, MixedElement):
        ismixed = True

    if ismixed:
        spacelist = []
        for elem in elem_lowest:
            spacelist.append(FunctionSpace(mesh, elem))
        space_lowest = MixedFunctionSpace(spacelist)
        # trialfuncs_lowest = TrialFunctions(space_lowest)
        testfuncs_lowest = TestFunctions(space_lowest)
        testfuncs = TestFunctions(testfunc.ufl_function_space())
        trialfuncs = TrialFunctions(trialfunc.ufl_function_space())
        rdict = {}
        for trialfunc, trialfunc_lowest in zip(trialfuncs, testfuncs_lowest):
            rdict[trialfunc] = trialfunc_lowest
        for testfunc, testfunc_lowest in zip(testfuncs, testfuncs_lowest):
            rdict[testfunc] = testfunc_lowest
        replacer = Replacer(rdict)

    else:
        space_lowest = FunctionSpace(mesh, elem_lowest)
        trialfunc_lowest = TrialFunction(space_lowest)
        testfunc_lowest = TestFunction(space_lowest)
        replacer = Replacer({testfunc: testfunc_lowest, trialfunc: trialfunc_lowest})
    newform = map_integrand_dags(replacer, form)

    return newform
