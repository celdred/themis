from ufl import Argument
from ufl.split_functions import split

from ufl.assertions import ufl_assert
from ufl.algorithms import extract_arguments, extract_coefficients

import ufl
from function import Function

class Argument(ufl.Argument):
	def function_space(self):
		return self.ufl_function_space()

####################
#THESE ARE FROM FIREDRAKE
#NEED ATTRIBUTION 
def TestFunction(function_space, part=None):
    """Build a test function on the specified function space.
    :arg function_space: the :class:`.FunctionSpace` to build the test
         function on.
    :kwarg part: optional index (mostly ignored)."""
    return Argument(function_space, 0, part=part)


def TrialFunction(function_space, part=None):
    """Build a trial function on the specified function space.
    :arg function_space: the :class:`.FunctionSpace` to build the trial
         function on.
    :kwarg part: optional index (mostly ignored)."""
    return Argument(function_space, 1, part=None)


def TestFunctions(function_space):
    """Return a tuple of test functions on the specified function space.
    :arg function_space: the :class:`.FunctionSpace` to build the test
         functions on.
    This returns ``len(function_space)`` test functions, which, if the
    function space is a :class:`.MixedFunctionSpace`, are indexed
    appropriately.
    """
    return split(TestFunction(function_space))


def TrialFunctions(function_space):
    """Return a tuple of trial functions on the specified function space.
    :arg function_space: the :class:`.FunctionSpace` to build the trial
         functions on.
    This returns ``len(function_space)`` trial functions, which, if the
    function space is a :class:`.MixedFunctionSpace`, are indexed
    appropriately.
    """
    return split(TrialFunction(function_space))


def derivative(form, u, du=None, coefficient_derivatives=None):
    """Compute the derivative of a form.
    Given a form, this computes its linearization with respect to the
    provided :class:`.Function`.  The resulting form has one
    additional :class:`Argument` in the same finite element space as
    the Function.
    :arg form: a :class:`~ufl.classes.Form` to compute the derivative of.
    :arg u: a :class:`.Function` to compute the derivative with
         respect to.
    :arg du: an optional :class:`Argument` to use as the replacement
         in the new form (constructed automatically if not provided).
    :arg coefficient_derivatives: an optional :class:`dict` to
         provide the derivative of a coefficient function.
    :raises ValueError: If any of the coefficients in ``form`` were
        obtained from ``u.split()``.  UFL doesn't notice that these
        are related to ``u`` and so therefore the derivative is
        wrong (instead one should have written ``split(u)``).
    See also :func:`ufl.derivative`.
    """
   #EVENTUALLY ADD THIS CHECKING BACK IN!
   # if set(extract_coefficients(form)) & set(u.split()):
    #    raise ValueError("Taking derivative of form wrt u, but form contains coefficients from u.split()."
     #                    "\nYou probably meant to write split(u) when defining your form.")
    
    if du is None:
        if isinstance(u, Function):
            V = u.function_space()
            args = form.arguments()
            number = max(a.number() for a in args) if args else -1
            du = Argument(V, number + 1)
        else:
            raise RuntimeError("Can't compute derivative for form")
    return ufl.derivative(form, u, du, coefficient_derivatives)


def adjoint(form, reordered_arguments=None):
    """UFL form operator:
    Given a combined bilinear form, compute the adjoint form by
    changing the ordering (number) of the test and trial functions.
    By default, new Argument objects will be created with
    opposite ordering. However, if the adjoint form is to
    be added to other forms later, their arguments must match.
    In that case, the user must provide a tuple reordered_arguments=(u2,v2).
    """

    # ufl.adjoint creates new Arguments if no reordered_arguments is
    # given.  To avoid that, always pass reordered_arguments with
    # firedrake.Argument objects.
    if reordered_arguments is None:
        v, u = extract_arguments(form)
        reordered_arguments = (Argument(u.function_space(),
                                        number=v.number(),
                                        part=v.part()),
                               Argument(v.function_space(),
                                        number=u.number(),
                                        part=u.part()))
    return ufl.adjoint(form, reordered_arguments)
    
#NEED A WRAPPER FOR CONSTANT/VECTOR CONSTANT/TENSOR CONSTANT
#MAYBE- SEE WHAT FIREDRAKE DOES HERE!
#CONSTANT IS A FIREDRAKE THING THAT NEEDS THINKING ABOUT..


