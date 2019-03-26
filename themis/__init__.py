

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
from ufl import *

from themis.assembly import *
from themis.bcs import *
from themis.checkpointer import *
from themis.codegenerator import *
from themis.constant import *
from themis.form import *
from themis.formmanipulation import *
from themis.function import *
from themis.functionspace import *
from themis.interpolator import *
from themis.mesh import *
from themis.norms import *
from themis.petscshim import *
from themis.plotting import *
from themis.project import *
from themis.quadrature import *
from themis.solver import *
from themis.themiselement import *
from themis.tsfc_interface import *
from themis.ufl_expr import *
from themis.utility_meshes import *
