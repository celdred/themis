

# Ensure petsc is initialised by us before anything else gets in there.
import themis.petscshim as petsc
del petsc

# UFL Exprs come with a custom __del__ method, but we hold references
# to them /everywhere/, some of which are circular (the Mesh object
# holds a ufl.Domain that references the Mesh).  The Python2 GC
# explicitly DOES NOT collect such reference cycles (even though it
# can deal with normal cycles).  Quoth the documentation:
#
#     Objects that have __del__() methods and are part of a reference
#     cycle cause the entire reference cycle to be uncollectable,
#     including objects not necessarily in the cycle but reachable
#     only from it.
#
# To get around this, since the default __del__ on Expr is just
# "pass", we just remove the method from the definition of Expr.
import ufl
try:
    del ufl.core.expr.Expr.__del__
except AttributeError:
    pass
del ufl
from ufl import *  # noqa: F401

from themis.assembly import *  # noqa: F401
from themis.bcs import *  # noqa: F401
from themis.checkpointer import *  # noqa: F401
from themis.codegenerator import *  # noqa: F401
from themis.constant import *  # noqa: F401
from themis.form import *  # noqa: F401
from themis.formmanipulation import *  # noqa: F401
from themis.function import *  # noqa: F401
from themis.functionspace import *  # noqa: F401
from themis.interpolator import *  # noqa: F401
from themis.mesh import *  # noqa: F401
from themis.norms import *  # noqa: F401
from themis.petscshim import *  # noqa: F401
from themis.plotting import *  # noqa: F401
from themis.project import *  # noqa: F401
from themis.quadrature import *  # noqa: F401
from themis.solver import *  # noqa: F401
from themis.themiselement import *  # noqa: F401
from themis.tsfc_interface import *  # noqa: F401
from themis.ufl_expr import *  # noqa: F401
from themis.utility_meshes import *  # noqa: F401
