Build and test COMET
======================

Once the :doc:`requirements` of LLVM/MLIR are met. One can proceed with building LLVM/MLIR and COMET as follows:

**Build and test MLIR**

:: 

   $ cd COMET
   $ mkdir llvm/build
   $ cd llvm/build
   $ cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=DEBUG
   $ ninja
   $ ninja check-mlir

**Build and test COMET**

::
  
   $ cd ../../
   $ mkdir build
   $ cd build
   $ cmake -G Ninja .. \
      -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
      -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=DEBUG
   $ ninja
   $ ninja check-comet
   $ ninja check-comet-integration # Run the integration tests.

The ``-DCMAKE_BUILD_TYPE=DEBUG`` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.
To get something that runs fast, use ``-DCMAKE_BUILD_TYPE=Release`` or
``-DCMAKE_BUILD_TYPE=RelWithDebInfo`` if you want to go fast and optionally if
you want debug info to go with it.  Release mode makes a very large difference
in performance.

.. autosummary::
   :toctree: generated

