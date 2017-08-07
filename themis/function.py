from ufl import Coefficient
from petscshim import PETSc
import numpy as np

	#NEED A GOOD DEFAULT NAME HERE!!!

class QuadCoefficient():
	def __init__(self,mesh,ctype,quad,name='xquad'):
		self.ctype = ctype
		self.npatches = mesh.npatches
		self.mesh = mesh
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
	
	def getarray(self,bi):
		arr = self.dms[bi].getVecArray(self.vecs[bi])[:]
		arr = np.reshape(arr,self.shapes[bi])
		return arr
	def name(self):
		return self._name

class SplitFunction():
	#NEED A GOOD DEFAULT NAME HERE!!!

	def __init__(self,space,name='xsplit',gvec=None,si=None,parentspace=None):
		self.space = space
		self._name = name
		self._si = si

		self._vector = gvec
		self._parentspace = parentspace

	def assign(self,v_in):
		#v_in must be either a Function or a SplitFunction, and it must have the same space as self
		if not isinstance(v_in,(Function,SplitFunction)):
			raise TypeError('Only Function or SplitFunction can be used as the source for an assignment')
		if not (self.space == v_in.space):
			raise TypeError('Source and target spaces for assignment must be the same')
		
		#2 cases: v_in is a Function, and v_in is a SplitFunction
		#in the latter case, need to extract subvector before copying
		
		#put v_in into the subvector
		subvec = self._vector.getSubVector(self._parentspace.get_field_gis(self._si))
		if isinstance(v_in,SplitFunction):
			vecin = v_in._vector.getSubVector(v_in._parentspace.get_field_gis(v_in._si))	
		else:
			vecin = v_in._vector		
		vecin.copy(result=subvec)
		if isinstance(v_in,SplitFunction):
			v_in._vector.restoreSubVector(v_in._parentspace.get_field_gis(v_in._si),vecin)			
		self._vector.restoreSubVector(self._parentspace.get_field_gis(self._si),subvec)
			
class Function(Coefficient):

	def assign(self,v_in):
		#v_in must be either a Function or a SplitFunction, and it must have the same space as self
		if not isinstance(v_in,(Function,SplitFunction)):
			raise TypeError('Only Function or SplitFunction can be used as the source for an assignment')
		if not (self.space == v_in.space):
			raise TypeError('Source and target spaces for assignment must be the same')
			
		#2 cases: v_in is a Function, and v_in is a SplitFunction
		#in the latter case, need to extract subvector before copying
		
		#put v_in into the subvector
		if isinstance(v_in,SplitFunction):
			vecin = v_in._vector.getSubVector(v_in._parentspace.get_field_gis(v_in._si))
		else:
			vecin = v_in._vector		
		vecin.copy(result=self._vector)
		if isinstance(v_in,SplitFunction):
			v_in._vector.restoreSubVector(v_in._parentspace.get_field_gis(v_in._si),vecin)			
		
	def function_space(self):
		return self.ufl_function_space()
	
	#EVENTUALLY THIS SHOULD BE SMART ENOUGH TO ONLY ACTUALLY SCATTER IF THE FIELD HAS BEEN UPDATED!
	#IE THROUGH AN ASSIGN OR A SOLVE!
	def scatter(self):
		self.space.get_composite_da().scatter(self._activevector,self._lvectors)

	#gvec,lvecs and si are used in the creation of split Functions
	
	#NEED A GOOD DEFAULT NAME HERE!!!
	def __init__(self,space,name='x'):
		with PETSc.Log.Stage(name + '_init'):
			
			Coefficient.__init__(self, space)
			self._name = name
			self.space = space
				
			with PETSc.Log.Event('create_global'):
				self._vector = self.space.get_composite_da().createGlobalVec()
				self._vector.set(0.0)
				self._activevector = self._vector
	
			with PETSc.Log.Event('create_local'):
				#create local vectors- one for EACH component on each PATCH
				self._lvectors = []
				for si in range(self.space.nspaces):
					for ci in range(self.space.get_space(si).ncomp):
						for bi in range(self.space.get_space(si).npatches):
							self._lvectors.append(self.space.get_space(si).get_da(ci,bi).createLocalVector())

			if self.space.nspaces > 1:
				self._split = tuple(SplitFunction(self.space.get_space(i), name="%s[%d]" % (self.name(), i), gvec = self._vector, si = i, parentspace=self.space) for i in range(self.space.nspaces))
			else:
				self._split = (self,)
				
	
	def get_lvec(self,si,ci,bi):
		soff = self.space.get_space_offset(si)
		coff = self.space.get_space(si).get_component_offset(ci)
		return self._lvectors[coff+soff+bi]
		
	def name(self):
		return self._name
	
	#These have no idea they are part of a mixed space, EXCEPT for si argument that is used internally to correctly offset into lvecs
	#and to extract relevant subvector of global vector
	def split(self):
		"""Extract any sub :class:`Function`\s defined on the component spaces
		of this this :class:`Function`'s :class:`.FunctionSpace`."""
		return self._split
