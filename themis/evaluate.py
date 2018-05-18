from petscshim import PETSc
from assembly import extract_fields,compile_functional
import numpy as np

class QuadCoefficient():
    def __init__(self, mesh, ctype, etype, evalfield, quad, name='xquad'):
        self.ctype = ctype
        self.etype = etype
        self.npatches = mesh.npatches
        self.mesh = mesh
        self.evalfield = evalfield
# NEED A GOOD DEFAULT NAME HERE!!!
        self._name = name

        self.vecs = []
        self.dms = []
        self.shapes = []
        for bi in range(self.npatches):
            dm = mesh.get_cell_da(bi)
            nquadlist = quad.get_nquad()
            nquad = np.prod(nquadlist)
            localnxs = mesh.get_local_nxny(bi)
            shape = list(localnxs)
            shape.append(nquadlist[0])
            shape.append(nquadlist[1])
            shape.append(nquadlist[2])

            if ctype == 'scalar':
                newdm = dm.duplicate(dof=nquad)
            if ctype == 'vector':
                newdm = dm.duplicate(dof=nquad*mesh.ndim)
                shape.append(mesh.ndim)
            if ctype == 'tensor':
                newdm = dm.duplicate(dof=nquad*mesh.ndims*mesh.ndim)
                shape.append(mesh.ndim)
                shape.append(mesh.ndim)

            self.dms.append(newdm)
            self.shapes.append(shape)
            self.vecs.append(self.dms[bi].createGlobalVector())

        self.evalkernel = EvalKernel(mesh,evalfield,quad,ctype,etype,name)
        
    def getarray(self, bi):
        arr = self.dms[bi].getVecArray(self.vecs[bi])[:]
        arr = np.reshape(arr, self.shapes[bi])
        return arr

    def name(self):
        return self._name
        
    def evaluate(self):
        EvaluateCoefficient(self,self.evalkernel)

class EvalKernel():
    def __init__(self,mesh,field, quad, ctype,etype,name):
        
        self.integral_type = 'cell'
        self.oriented = False
        self.coefficient_numbers = [0, ]
        self.coefficient_map = [0,] 
        self.zero = False
        self.assemblycompiled = False
        self.evaluate = True
        
        self.ctype = ctype
        self.etype = etype
        self.name = name
        self.mesh = mesh
        self.coefficients = [field,]  
        self.quad = quad
        
def EvaluateCoefficient(coefficient, evalkernel):
    with PETSc.Log.Stage(coefficient.name() + '_evaluate'):

        #compile the kernel
        with PETSc.Log.Event('compile'):
            if not evalkernel.assemblycompiled:
                compile_functional(evalkernel, None, None, coefficient.mesh)

        # scatter fields into local vecs
        with PETSc.Log.Event('extract'):
            extract_fields(evalkernel)

        # evaluate
        with PETSc.Log.Event('evaluate'):
            for da, assemblefunc in zip(evalkernel.dalist, evalkernel.assemblyfunc_list):
                # BROKEN FOR MULTIPATCH- COEFFICIENT DMS/VEC AND FIELD ARGS LIST NEEDS A BI INDEX
                assemblefunc([da, ] +  [coefficient.dms[0],coefficient.vecs[0]] + evalkernel.fieldargs_list, evalkernel.constantargs_list)


