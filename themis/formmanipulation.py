import numpy
from themis.ufl_expr import Argument
from ufl import as_vector
from ufl.classes import Zero
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.map_dag import MultiFunction
from themis.functionspace import FunctionSpace
from ufl import MixedElement

__all__ = ["FormSplitter", "split_form"]


class FormSplitter(MultiFunction):

    def split(self, form, ix=0, iy=0):
        # Remember which block to extract
        self.idx = [ix, iy]
        args = form.arguments()
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):  # not mixed, just return self
            # SHOULD/CAN THESE BE RESTORED?
            # WHAT EXACTLY WERE THEY CHECKING?
            # assert (len(idx) == 1 for idx in self.blocks.values())
            # assert (idx[0] == 0 for idx in self.blocks.values())
            return form
        return map_integrand_dags(self, form)

    def argument(self, obj):
        Q = obj.ufl_function_space()
        dom = Q.ufl_domain()
        sub_elements = obj.ufl_element().sub_elements()

        # If not a mixed element, do nothing
        if not isinstance(obj.ufl_element(), MixedElement):
            return obj

    # Split into sub-elements, creating appropriate space for each
        args = []
        for i, sub_elem in enumerate(sub_elements):
            Q_i = FunctionSpace(dom, sub_elem)
            a = Argument(Q_i, obj.number(), part=obj.part())

            indices = [()]
            for m in a.ufl_shape:
                indices = [(k + (j,)) for k in indices for j in range(m)]

            if (i == self.idx[obj.number()]):
                args += [a[j] for j in indices]
            else:
                args += [Zero() for j in indices]

        return as_vector(args)

    def multi_index(self, obj):
        return obj

    expr = MultiFunction.reuse_if_untouched


def split_form(form):
    args = form.arguments()
    shape = tuple(len(a.function_space()) for a in args)
    forms = []
    splitter = FormSplitter()
    for idx in numpy.ndindex(shape):
        f = splitter.split(form, *idx)
        if len(f.integrals()) > 0:
            forms.append((idx, f))
    return tuple(forms)
