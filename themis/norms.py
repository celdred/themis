from themis.function import Function
from ufl import inner, div, grad, curl, sqrt, dx
from themis.form import ZeroForm

__all__ = ["errornorm", "norm"]


def errornorm(u, uh, norm_type="L2"):
    """Compute the error :math:`e = u - u_h` in the specified norm.

    :arg u: a :class:`.Function` or UFL expression containing an "exact" solution
    :arg uh: a :class:`.Function` containing the approximate solution
    :arg norm_type: the type of norm to compute, see :func:`.norm` for
    details of supported norm types.
    """

    urank = len(u.ufl_shape)
    uhrank = len(uh.ufl_shape)

    if urank != uhrank:
        raise RuntimeError("Mismatching rank between u and uh")

    if not isinstance(uh, Function):
        raise ValueError("uh should be a Function, is a %r", type(uh))

    return norm(u - uh, norm_type=norm_type)


def norm(v, norm_type="L2", degree=None):
    """Compute the norm of ``v``.

    :arg v: a ufl expression (:class:`~.ufl.classes.Expr`) to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
             options.

    Available norm types are:

    * L2

       .. math::

              ||v||_{L^2}^2 = \int (v, v) \mathrm{d}x

    * H1

       .. math::

              ||v||_{H^1}^2 = \int (v, v) + (\\nabla v, \\nabla v) \mathrm{d}x

    * Hdiv

       .. math::

              ||v||_{H_\mathrm{div}}^2 = \int (v, v) + (\\nabla\cdot v, \\nabla \cdot v) \mathrm{d}x

    * Hcurl

       .. math::

              ||v||_{H_\mathrm{curl}}^2 = \int (v, v) + (\\nabla \wedge v, \\nabla \wedge v) \mathrm{d}x
    """

    typ = norm_type.lower()
    if typ == 'l2':
        form = inner(v, v)*dx
    elif typ == 'h1':
        form = inner(v, v)*dx + inner(grad(v), grad(v))*dx
    elif typ == "hdiv":
        form = inner(v, v)*dx + div(v)*div(v)*dx
    elif typ == "hcurl":
        form = inner(v, v)*dx + inner(curl(v), curl(v))*dx
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    normform = ZeroForm(form)
    return sqrt(normform.assembleform())
