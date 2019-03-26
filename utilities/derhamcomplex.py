
from interop import FiniteElement, TensorProductElement, VectorElement, HDivElement, HCurlElement
from interop import quadrilateral, interval, triangle, hexahedron

__all__ = ['create_complex', ]

def _quad(order, variant):
    h1elem = FiniteElement("Q", quadrilateral, order, variant=variant)
    l2elem = FiniteElement("DQ", quadrilateral, order-1, variant=variant)
    hdivelem = FiniteElement('RTCF', quadrilateral, order, variant=variant)
    hcurlelem = FiniteElement('RTCE', quadrilateral, order, variant=variant)
    return h1elem, l2elem, hdivelem, hcurlelem


def _tri(order, velocityspace):
    # CGk - RTk-1 - DGk-1 on triangles (order = k)
    if velocityspace == 'RT':
        h1elem = FiniteElement("CG", triangle, order)
        l2elem = FiniteElement('DG', triangle, order-1)
        # RT gives spurious inertia-gravity waves
        hdivelem = FiniteElement('RTF', triangle, order)
        hcurlelem = FiniteElement('RTE', triangle, order)

    # CGk - BDMk-1 - DGk-2 on triangles (order = k-1)
    if velocityspace == 'BDM':
        h1elem = FiniteElement("CG", triangle, order+1)
        l2elem = FiniteElement("DG", triangle, order-1)
        # BDM gives spurious Rossby waves
        hdivelem = FiniteElement("BDMF", triangle, order)
        hcurlelem = FiniteElement("BDME", triangle, order)

    # CG2+B3 - BDFM1 - DG1 on triangles
    if velocityspace == 'BDFM':
        if not (order == 2):
            raise ValueError('BDFM space is only supported for n=2')
        h1elem = FiniteElement("CG", triangle, order) + FiniteElement("Bubble", triangle, order + 1)
        l2elem = FiniteElement("DG", triangle, order-1)
        # Note that n=2 is the lowest order element...
        hdivelem = FiniteElement("BDFM", triangle, order)
        # WHAT IS THE CORRESPONDING HCURL ELEMENT?
        # IS THERE ONE?
        hcurlelem = None
    return h1elem, l2elem, hdivelem, hcurlelem


def _interval(order, variant):
    h1elem = FiniteElement("CG", interval, order, variant=variant)
    l2elem = FiniteElement("DG", interval, order-1, variant=variant)
    return h1elem, l2elem


def _hex(order, variant):
    h1elem = FiniteElement("Q", hexahedron, order, variant=variant)
    l2elem = FiniteElement("DQ", hexahedron, order-1, variant=variant)
    hdivelem = FiniteElement("NCF", hexahedron, order, variant=variant)
    hcurlelem = FiniteElement("NCE", hexahedron, order, variant=variant)
    return h1elem, l2elem, hdivelem, hcurlelem

# ADD SUPPORT FOR CR and SE AS WELL...
# if velocityspace == 'SE':
    #elemdict['h1'] = FiniteElement("CG", interval, order,variant='spectral')
# if velocityspace == 'NC':
    #elemdict['l2'] = FiniteElement("CR", triangle, order)
# MAYBE? MAYBE JUST DO THEM AS SEPARATE ELEMENTS..
# WHAT ABOUT DG GLL?


def create_complex(cell, velocityspace, variant, order, vorder=None):
    vorder = vorder or order
    if variant == 'none':  # this ensures Firedrake works properly
        variant = None

    if cell == 'interval':
        h1elem, l2elem = _interval(order, variant)
        hdivelem = VectorElement(h1elem, dim=1)
        hcurlelem = VectorElement(l2elem, dim=1)
        cpelem = None
    if cell == 'quad':
        h1elem, l2elem, hdivelem, hcurlelem = _quad(order, variant)
        cpelem = None
    if cell == 'tri':
        if not variant is None:
            raise ValueError("variants are not support on triangles")
        h1elem, l2elem, hdivelem, hcurlelem = _tri(order, velocityspace)
        cpelem = None
    if cell == 'hex':
        h1elem, l2elem, hdivelem, hcurlelem = _quad(order, variant)
    if cell in ['tpquad', 'tphex', 'tptri']:
        if cell == 'tpquad':
            h1elem2D, l2elem2D = _interval(vorder, variant)
            hdivelem2D, hcurlelem2D = _interval(vorder, variant)
        if cell == 'tphex':
            h1elem2D, l2elem2D, hdivelem2D, hcurlelem2D, _ = _quad(order, variant)
        if cell == 'tptri':
            if not variant is None:
                raise ValueError("variants are not support on triangles")
            h1elem2D, l2elem2D, hdivelem2D, hcurlelem2D, _ = _tri(order, velocityspace)
        h1elem1D, l2elem1D = _interval(vorder, variant)
        hdivelem = HDivElement(TensorProductElement(hdivelem2D, l2elem1D)) + HDivElement(TensorProductElement(l2elem2D, h1elem1D))
        hcurlelem = HCurlElement(TensorProductElement(hcurlelem2D, h1elem1D)) + HCurlElement(TensorProductElement(h1elem2D, l2elem1D))
        h1elem = TensorProductElement(h1elem2D, h1elem1D)
        l2elem = TensorProductElement(l2elem2D, l2elem1D)
        cpelem = TensorProductElement(l2elem2D, h1elem1D)

    return {'h1': h1elem, 'l2': l2elem, 'hdiv': hdivelem, 'hcurl': hcurlelem, 'cp': cpelem}
