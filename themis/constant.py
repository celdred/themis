import numpy as np
import ufl

# ADD FIREDRAKE ATTRIBUTION

__all__ = ["Constant", ]


def _globalify(value):
    data = np.array(value, dtype=np.float64)
    shape = data.shape
    rank = len(shape)
    if rank in [0,1,2]:
        dat = data
    else:
        raise RuntimeError("Don't know how to make Constant from data with rank %d" % rank)
    return dat, rank, shape

# ADD FIREDRAKE ATTRIBUTION


class Constant(ufl.Coefficient):

    """A "constant" coefficient

    A :class:`Constant` takes one value over the whole
    :func:`~.Mesh`. The advantage of using a :class:`Constant` in a
    form rather than a literal value is that the constant will be
    passed as an argument to the generated kernel which avoids the
    need to recompile the kernel if the form is assembled for a
    different value of the constant.

    :arg value: the value of the constant.  May either be a scalar, an
             iterable of values (for a vector-valued constant), or an iterable
             of iterables (or numpy array with 2-dimensional shape) for a
             tensor-valued constant.
    """

    def __init__(self, value):

        self.dat, rank, shape = _globalify(value)

        cell = None
        domain = None
        if rank == 0:
            e = ufl.FiniteElement("Real", cell, 0)
        elif rank == 1:
            e = ufl.VectorElement("Real", cell, 0, shape[0])
        elif rank == 2:
            e = ufl.TensorElement("Real", cell, 0, shape=shape)

        fs = ufl.FunctionSpace(domain, e)
        super(Constant, self).__init__(fs)
        self._repr = 'Constant(%r, %r)' % (self.ufl_element(), self.count())

# THIS BREAKS FOR NON-SCALAR CONSTANTS, WHICH I THINK IS THE CORRECT BEHAVIOUR ANYWAYS?
    def __float__(self):
        return float(self.dat)

    def name(self):
        # return self.__str__().translate(None,'{}')
        return self.__str__().translate(str.maketrans('', '', '{}'))

    def evaluate(self, x, mapping, component, index_values):
        """Return the evaluation of this :class:`Constant`.

        :arg x: The coordinate to evaluate at (ignored).
        :arg mapping: A mapping (ignored).
        :arg component: The requested component of the constant (may
        be ``None`` or ``()`` to obtain all components).
        :arg index_values: ignored.
        """
        if component in ((), None):
            if self.ufl_shape is ():
                return self.dat[0]
            return self.dat
        return self.dat[component]

    def function_space(self):
        """Return a null function space."""
        return None

    def split(self):
        return (self,)

    def assign(self, value):
        """Set the value of this constant.
        :arg value: A value of the appropriate shape"""
        self.dat, _, _ = _globalify(value)

    def __iadd__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __isub__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __imul__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __idiv__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")
