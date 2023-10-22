Build and test COMET
======================

Once the :doc:`requirements` of LLVM/MLIR are met. One can proceed with building LLVM/MLIR and COMET as follows:

**Build and test LLVM/MLIR**

::

   $ cd $COMET_SRC
   $ mkdir llvm/build
   $ cd llvm/build
   $ cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;openmp;clang" \
      -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
      -DCMAKE_OSX_ARCHITECTURES="arm64" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release
   $ ninja
   $ ninja check-mlir

**Apply BLIS patch to meet COMET requirements**

::

   $ cd $COMET_SRC
   $ patch -s -p0 < comet-blis.patch

**Build and test BLIS**

::
   $ cd $COMET_SRC
   $ cd blis
   $ ./configure --prefix=$COMET_SRC/install auto
   $ make [-j]
   $ make check [-j]
   $ make install [-j]

**Build and test COMET**

::
  
   $ cd $COMET_SRC
   $ mkdir build
   $ cd build
   $ cmake -G Ninja .. \
      -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
      -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release
   $ ninja
   $ ninja check-comet-integration # Run the integration tests.

The ``-DCMAKE_BUILD_TYPE=DEBUG`` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.
To get something that runs fast, use ``-DCMAKE_BUILD_TYPE=Release`` or
``-DCMAKE_BUILD_TYPE=RelWithDebInfo`` if you want to go fast and optionally if
you want debug info to go with it. Release mode makes a very large difference
in performance.

.. autosummary::
   :toctree: generated

