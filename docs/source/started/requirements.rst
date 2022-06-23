Requirements
============

COMET compiler requires a specific version of LLVM/MLIR that has been included as git submodule inside the COMET repo.
Therefore, there is no need to do a separate clone of the LLVM/MLIR repo. 
Before proceeding with the build and testing of MLIR/LLVM and COMET, 
please install `CMake (3.13.4 or later) <https://cmake.org/download>`_,
`Ninja (1.5 or later) <https://ninja-build.org/>`_, 
a C++ compiler toolchain as `mentioned here <https://llvm.org/docs/GettingStarted.html#requirements>`_,
and `Python3 (3.6 or later) <https://www.python.org/downloads/>`_.

**Check out COMET repo**: COMET contains LLVM/MLIR as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support COMET.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

::

   $ git clone https://github.com/pnnl/COMET.git
   $ cd COMET
   $ git submodule init
   $ git submodule update

*Note*: The repository is set up so that git submodule update performs a
shallow clone, meaning it downloads just enough of the LLVM/MLIR repository to check
out the currently specified commit.

.. autosummary::
   :toctree: generated

