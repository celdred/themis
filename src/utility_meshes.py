from mesh import SingleBlockMesh

from ufl import FiniteElement,TensorProductElement,interval,VectorElement
from functionspace import FunctionSpace
from function import Function
import numpy as np
	
def create_box_mesh(nxs,lxs,pxs,bcs):
	assert(len(nxs) == len(lxs))
	assert(len(nxs) == len(pxs))
	assert(len(nxs) == len(bcs))
	
	pxs = np.array(pxs,dtype=np.float32)
	lxs = np.array(lxs,dtype=np.float32)
	nxs = np.array(nxs,dtype=np.int32)
	dxs = lxs/nxs
	
	mesh = SingleBlockMesh(nxs,bcs)
	
	coordsdm = mesh.coordinates.space.get_da(0,0)
	coordsarr = coordsdm.getVecArray(mesh.coordinates._vector)[:]

	if len(dxs) == 1: coordsarr = np.expand_dims(coordsarr,axis=1)
	for i in range(len(dxs)):
		coordsarr[...,i] = coordsarr[...,i] * dxs[i]/2. + pxs[i]
	
	mesh.coordinates.scatter()

	return mesh

def PeriodicIntervalMesh(ncells, length_or_left, right=None):
	"""
	Generate a uniform mesh of an interval.

	:arg ncells: The number of the cells over the interval.
	:arg length_or_left: The length of the interval (if ``right``
		 is not provided) or else the left hand boundary point.
	:arg right: (optional) position of the right
		 boundary point (in which case ``length_or_left`` should
		 be the left boundary point).
		 
	Creates a periodic domain with extent [0,length] OR extent [left,right]
	"""

	if right is None:
		left = 0
		right = length_or_left
	else:
		left = length_or_left

	if ncells <= 0 or ncells % 1:
		raise ValueError("Number of cells must be a postive integer")
	length = right - left
	if length < 0:
		raise ValueError("Requested mesh has negative length")

	nxs = [ncells,]
	lxs = [length,]
	pxs = [length/2.0,]
	bcs = ['periodic',]
	return create_box_mesh(nxs,lxs,pxs,bcs)
	

def PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="both",quadrilateral=True):
	"""Generate a periodic rectangular mesh

	:arg nx: The number of cells in the x direction
	:arg ny: The number of cells in the y direction
	:arg Lx: The extent in the x direction
	:arg Ly: The extent in the y direction
	:arg direction: The direction of the periodicity, one of
		``"both"``, ``"x"`` or ``"y"``.

	Creates a domain with extent [0,Lx] x [0,Ly]

	"""

	for n in (nx, ny):
		if n <= 0 or n % 1:
			raise ValueError("Number of cells must be a postive integer")

	nxs = [nx,ny]
	lxs = [Lx,Ly]
	pxs = [Lx/2.0,Ly/2.0]
	if direction == 'both':
		bcs = ['periodic','periodic']
	if direction == 'x':
		bcs = ['periodic','nonperiodic']
	if direction == 'y':
		bcs = ['nonperiodic','periodic']
	return create_box_mesh(nxs,lxs,pxs,bcs)
	
def IntervalMesh(ncells, length_or_left, right=None):
	"""
	Generate a uniform mesh of an interval.

	:arg ncells: The number of the cells over the interval.
	:arg length_or_left: The length of the interval (if ``right``
	 is not provided) or else the left hand boundary point.
	:arg right: (optional) position of the right
	 boundary point (in which case ``length_or_left`` should
	 be the left boundary point).
	 
	Creates a domain with extent [0,length] OR extent [left,right]
	"""
	if right is None:
		left = 0
		right = length_or_left
	else:
		left = length_or_left

	if ncells <= 0 or ncells % 1:
		raise ValueError("Number of cells must be a postive integer")
	length = right - left
	if length < 0:
		raise ValueError("Requested mesh has negative length")

	nxs = [ncells,]
	lxs = [length,]
	pxs = [length/2.0,]
	bcs = ['nonperiodic',]
	return create_box_mesh(nxs,lxs,pxs,bcs)

def RectangleMesh(nx, ny, Lx, Ly,quadrilateral=True):
	"""Generate a rectangular mesh

	:arg nx: The number of cells in the x direction
	:arg ny: The number of cells in the y direction
	:arg Lx: The extent in the x direction
	:arg Ly: The extent in the y direction

	Creates a domain with extent [0,Lx] x [0,Ly]
	"""

	for n in (nx, ny):
		if n <= 0 or n % 1:
			raise ValueError("Number of cells must be a postive integer")

	nxs = [nx,ny]
	lxs = [Lx,Ly]
	pxs = [Lx/2.0,Ly/2.0]
	bcs = ['nonperiodic','nonperiodic']
	return create_box_mesh(nxs,lxs,pxs,bcs)
