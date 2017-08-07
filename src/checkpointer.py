from petscshim import PETSc
from function import Function,SplitFunction,QuadCoefficient
from firedrake import hdf5interface as h5i
import numpy as np

FILE_READ = PETSc.Viewer.Mode.READ
"""Open a checkpoint file for reading.  Raises an error if file does not exist."""

FILE_CREATE = PETSc.Viewer.Mode.WRITE
"""Create a checkpoint file.  Truncates the file if it exists."""

FILE_UPDATE = PETSc.Viewer.Mode.APPEND
"""Open a checkpoint file for updating.  Creates the file if it does not exist, providing both read and write access."""


#ADD FIREDRAKE ATTRIBUTION
class Checkpoint():
	def __init__(self,name,mode=FILE_UPDATE):
		
		self.mode = mode
		self._time = None
		self._tidx = -1
        
		import os
		exists = os.path.exists(name + '.h5')
		if self.mode == FILE_READ and not exists:
			raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
		mode = self.mode
		if mode == FILE_UPDATE and not exists:
			mode = FILE_CREATE
		self.viewer = PETSc.ViewerHDF5().create(name + '.h5', mode=mode,comm=PETSc.COMM_WORLD)

		self.h5file = h5i.get_h5py_file(self.viewer)
	
	def store_quad(self,quadcoeff,name=None):
		if self.mode is FILE_READ:
			raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
		if not isinstance(quadcoeff, QuadCoefficient):
			raise ValueError("Can only store QuadCoefficient")

		name = name or quadcoeff.name()
		group = self._get_data_group()
		self._write_timestep_attr(group)
		self.viewer.pushGroup(group)
		for bi in range(quadcoeff.npatches):
			quadcoeff.vecs[bi].setName(name + str(bi))
			self.viewer.view(obj=quadcoeff.vecs[bi])
		self.viewer.popGroup()
		
	def load_quad(self,quadcoeff,name=None):
		if not isinstance(quadcoeff, QuadCoefficient):
			raise ValueError("Can only load QuadCoefficient")

		group = self._get_data_group()
		self.viewer.pushGroup(group)
		name = name or quadcoeff.name()
		for bi in range(quadcoeff.npatches):
			quadcoeff.vecs[bi].setName(name + str(bi))
			quadcoeff.vecs[bi].load(self.viewer)
		self.viewer.popGroup()

		
	def store(self,field,name=None):
		if self.mode is FILE_READ:
			raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
		if not isinstance(field, Function):
			raise ValueError("Can only store functions")

		name = name or field.name()
		group = self._get_data_group()
		self._write_timestep_attr(group)

		self.viewer.pushGroup(group)
		with field.space.get_composite_da().getAccess(field._vector) as splitglobalvec:
			silist = range(field.space.nspaces)
			if isinstance(field,SplitFunction):
				silist = [field._si,]
			for si in silist:
				soff = field.space.get_space_offset(si)
				for ci in range(field.space.get_space(si).ncomp):
					coff = field.space.get_space(si).get_component_offset(ci)
					for bi in range(field.space.get_space(si).npatches):
						splitglobalvec[bi+soff+coff].setName(name + '_' + str(si) + '_' + str(ci) + '_' + str(bi))
						self.viewer.view(obj=splitglobalvec[bi+soff+coff])
		self.viewer.popGroup()
	
	def load(self,field,name=None):
		if not isinstance(field, Function):
			raise ValueError("Can only load functions")

		name = name or field.name()
		group = self._get_data_group()
		self.viewer.pushGroup(group)
		with field.space.get_composite_da().getAccess(field._vector) as splitglobalvec:
			silist = range(field.space.nspaces)
			if isinstance(field,SplitFunction):
				silist = [field._si,]
			for si in silist:
				soff = field.space.get_space_offset(si)
				for ci in range(field.space.get_space(si).ncomp):
					coff = field.space.get_space(si).get_component_offset(ci)
					for bi in range(field.space.get_space(si).npatches):
						splitglobalvec[bi+soff+coff].setName(name + '_' + str(si) + '_' + str(ci) + '_' + str(bi))
						splitglobalvec[bi+soff+coff].load(self.viewer)
		self.viewer.popGroup()

	def destroy(self):
		#destroy PETSc viewer
		self.viewer.destroy()
		self.h5file.flush()

	def write_attribute(self, obj, name, val):
		"""Set an HDF5 attribute on a specified data object.
		:arg obj: The path to the data object.
		:arg name: The name of the attribute.
		:arg val: The attribute value.
		Raises :exc:`~.exceptions.AttributeError` if writing the attribute fails.
		"""
		try:
			self.h5file[obj].attrs[name] = val
		except KeyError:
			raise AttributeError("Object '%s' not found" % obj)

	def _get_data_group(self):
		"""Return the group name for function data.
		If a timestep is set, this incorporates the current timestep
		index.  See :meth:`.set_timestep`."""
		if self._time is not None:
			return "/fields/%d" % self._tidx
		return "/fields"
	
	def set_timestep(self, t, idx=None):
		"""Set the timestep for output.
		:arg t: The timestep value.
		:arg idx: An optional timestep index to use, otherwise an
		internal index is used, incremented by 1 every time
		:meth:`set_timestep` is called.
		"""
		if idx is not None:
			self._tidx = idx
		else:
			self._tidx += 1
			self._time = t
		if self.mode == FILE_READ:
			return
		indices = self.read_attribute("/", "stored_time_indices", [])
		new_indices = np.concatenate((indices, [self._tidx]))
		self.write_attribute("/", "stored_time_indices", new_indices)
		steps = self.read_attribute("/", "stored_time_steps", [])
		new_steps = np.concatenate((steps, [self._time]))
		self.write_attribute("/", "stored_time_steps", new_steps)
	
	def get_timesteps(self):
		"""Return all the time steps (and time indices) in the current
		checkpoint file.
		This is useful when reloading from a checkpoint file that
		contains multiple timesteps and one wishes to determine the
		final available timestep in the file."""
		indices = self.read_attribute("/", "stored_time_indices", [])
		steps = self.read_attribute("/", "stored_time_steps", [])
		return steps, indices

	def _write_timestep_attr(self, group):
		"""Write the current timestep value (if it exists) to the
		specified group."""
		if self._time is not None:
			self.h5file.require_group(group)
			self.write_attribute(group, "timestep", self._time)

	def read_attribute(self, obj, name, default=None):
		"""Read an HDF5 attribute on a specified data object.
		:arg obj: The path to the data object.
		:arg name: The name of the attribute.
		:arg default: Optional default value to return.  If not
			 provided an :exc:`~.exceptions.AttributeError` is raised if the
			 attribute does not exist.
		"""
		try:
			return self.h5file[obj].attrs[name]
		except KeyError:
			if default is not None:
				return default
			raise AttributeError("Attribute '%s' on '%s' not found" % (name, obj))

	def close(self):
		"""Close the checkpoint file (flushing any pending writes)"""
		self.h5file.flush()
		self.viewer.destroy()
