Requirements
============

COMET compiler requires a specific version of LLVM/MLIR that has been included as git submodule inside the COMET repo.
Therefore, there is no need to do a separate clone of the LLVM/MLIR repo. 
Before proceeding with the build and testing of MLIR/LLVM and COMET, 
please install `CMake <https://cmake.org/>`_, `Ninja <https://ninja-build.org/>`_ and a C++ compiler toolchain as `mentioned here <https://llvm.org/docs/GettingStarted.html#requirements>`_.

**Check out COMET repos**: COMET contains LLVM/MLIR as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support COMET.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

::

   $ git clone https://github.com/pnnl/COMET.git
   $ cd comet
   $ git submodule init
   $ git submodule update

*Note*: The repository is set up so that git submodule update performs a
shallow clone, meaning it downloads just enough of the LLVM/MLIR repository to check
out the currently specified commit. If you wish to work with the full history of
the LLVM/MLIR repository, you can manually "unshallow" the submodule:

::

   $ cd llvm
   $ git fetch --unshallow
 

.. autosummary::
   :toctree: generated
