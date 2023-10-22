Requirements
============

COMET compiler requires a specific version of LLVM/MLIR that has been included as git submodule inside the COMET repo.

**Install Dependencies** 

To install COMET and LLVM/MLIR, the following dependencies need to be installed:
* `CMake (3.25 or later) <https://cmake.org/download>`_,
* `Ninja (1.5 or later) <https://ninja-build.org/>`_, 
* C++ compiler toolchain as `mentioned here <https://llvm.org/docs/GettingStarted.html#requirements>`_,
* `Python3 (3.9 or later) <https://www.python.org/downloads/>`_.

**Get submodules required for COMET** 

COMET contains LLVM and blis as a git submodule. The LLVM repo here includes staged changes to MLIR which may be necessary to support COMET. It also represents the version of LLVM that has been tested. MLIR is still changing relatively rapidly, so feel free to use the current version of LLVM, but APIs may have changed. BLIS is an award-winning portable software framework for instantiating high-performance BLAS-like dense linear algebra libraries. COMET generates a call to BLIS microkernel after some optimizations.

::

   $ git clone https://github.com/pnnl/COMET.git
   $ export COMET_SRC=`pwd`/COMET
   $ cd $COMET_SRC
   $ git submodule init
   $ git submodule update

*Note*: The repository is set up so that git submodule update performs a
shallow clone, meaning it downloads just enough of the LLVM/MLIR repository to check out the currently specified commit.

.. autosummary::
   :toctree: generated

