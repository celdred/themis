from __future__ import absolute_import, print_function, division

from distutils.core import setup
import sys

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)


setup(name="Themis",
      version="0.1",
      description="patch-structured Galerkin framework",
      author="Chris Eldred and others",
      author_email="chris.eldred@gmail.com",
      url="",
      license="",
      packages=["themis"])
