import os
import petsc4py
import numpy

include_dirs = [os.path.join(os.getenv('PETSC_DIR'), 'include'),
                petsc4py.get_include(), numpy.get_include(),
                # May need to add extras here depending on your environment
                '/usr/lib/openmpi/include', '/usr/lib/openmpi/include/openmpi',
                ]

swig_include_dirs = [petsc4py.get_include()]
library_dirs = [os.path.join(os.getenv('PETSC_DIR'), 'lib')]  # os.getenv('PETSC_ARCH')
libraries = ['petsc']
