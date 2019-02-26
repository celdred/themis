# Themis

# Installation

First, install Firedrake using the instructions at https://www.firedrakeproject.org/download.html. Then, with the firedrake venv active, type:

```bash
firedrake-update - -install git+ssh: // github.com/celdred/themis.git
```

# README #

Themis is a PETSc-based software framework for parallel, high-performance discretization of variational forms (and solution of systems of equations involving them) using tensor-product Galerkin methods on block-structured grids. It is intended to enable a rapid cycle of prototyping and experimentation, accelerating both the development of new numerical methods and scientific models that incorporate them. It does this through leveraging a range of existing software packages (PETSc, petsc4py, UFL, TSFC) and restricting the focus to a useful subset of methods (tensor-product Galerkin). 

User programs are written using the high-level UFL language to express variational forms, which are then compiled by TSFC to produce optimized local assembly kernels. Themis shares high-level design and goals with the FEniCS and Firedrake projects, and shared the same problem solving lanuage as Firedrake. In fact, valid Themis programs are also valid Firedrake programs (provided the import lines at the top are changed).

The majority of the code is written in Python, with performance critical parts either occuring natively in C through the petsc4py interface to PETSc (or in Numpy) or automatically generated in C using TSFC in conjuction with UFL.

As it currently stands, Themis is in ALPHA and there is little documentation available. However, many examples can be found under the tests directory, including 1D/2D/3D standard and mixed Helmholtz, Symmetric Interior Penalty DG, Vector Laplacian using mixed elements, and Bratu using H1 elements. A key design goal is that valid Themis programs are also valid Firedrake programs, so additional examples can found at https://www.firedrakeproject.org/. Additionally, a framework for quasi-Hamiltonian numerical models (based on discretizing the Poisson bracket and Hamiltonian rather than the equations of motion) termed Mistral is under development using Themis, initially targeted towards atmospheric dynamical cores. Please send us an email if you would like access to the repository.

Our primary scientific focus is atmospheric dynamical cores, and our choice of applications reflects this. We would welcome contributions from other fields!

### Existing Capabilities ###
    
* Support for single-block, structured grids in 1, 2 and 3 dimensions
* Automated generation and compilation of assembly code or matrix-free operator evaluation using UFL and TSFC
* Arbitrary mappings between physical and reference space
* Support for mixed, vector and standard tensor-product Galerkin function spaces using the mimetic Galerkin difference $MGD_n$ and $Q_r^- \Lambda^K$ families 
* Arbitrary choice of $H^1$ and $L_2$ basis functions in 1D (extended to nD via tensor-products) for the $Q_r^- \Lambda^K$ family, including Bernstein polynomials and defining a compatible $L_2$ basis from an $H_1$ basis (gives the mimetic spectral element family)
* Variational forms of degree 0, 1 and 2 and fields defined on those function spaces
* Solution of linear and nonlinear variational problems involving forms and fields on those function spaces
* Support for essential, natural and periodic boundary conditions
* Access to symbolic representation of basis functions (useful for calculation of dispersion relationships, amongst other applications)
* Parallelism through MPI

### Planned Future Capabilities ###

* New compatible Galerkin family: isogeometric differential forms
* Multi-block, structured grids
* Geometric multigrid
* Support for composable preconditioners
* Mimetic Galerkin Difference specific preconditioners
* Isogeometric analyis specific preconditioners
* Improved support for matrix-free operator evaluation
* Incorporation of firedrake-adjoint

### I want to work on Themis.../I have this great contribution... ###

For contributions, please submit a pull request, and if it look good we will add it in!

### Help! Themis is not working... ###

Please open a bug report with any issues you encounter. Unforunately, we can only offer very limited support at this time. Every effort will be made to reply in a timely manner, but don't be alarmed if it takes a few days to receive a response! 

### Known Issues ###

### License ###
Themis is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Themis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with Themis. If not, see <http://www.gnu.org/licenses/>.
