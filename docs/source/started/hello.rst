COMET DSL Hello World
=====================

This section will go over steps to compile your first COMET DSL program, once COMET has been successfully built (see: :doc:`install`).

*myFirst.ta*
::

   def main() {
     # IndexLabel declarations
     IndexLabel [i] = [4];                 # static index label
     IndexLabel [j] = [4];
     IndexLabel [k] = [4];

     # Tensor declarations
     Tensor<double> A([i, j], {Dense});    # declare a dense tensor
     Tensor<double> B([j, k], {Dense});
     Tensor<double> C([i, k], {Dense});

     # Tensor Fill operation
     A[i, j] = 2.4;                        # initialization
     B[j, k] = 3.2;
     C[i, k] = 0.0;                        # all values are initialized to 0

     # Tensor Contraction
     C[i, k] = A[i, j] * B[j, k];          # perform matrix multiplication

     print(C);                             # print the matrix
   } 

Convert *TA dialect* to *IT dialect* using ``--convert-ta-to-it`` pass, and then to *loops (SCF dialect)* using ``--convert-to-loops`` pass. These passes are available in ``comet-opt`` that should be located in ``build/bin``:
::

   $ comet-opt --convert-ta-to-it --convert-to-loops myFirst.ta &> myFirst.mlir
   
Convert *loops (SCF dialect)* to *std dialect* using ``--convert-scf-to-std`` pass and then *std* to *LLVM dialect* using ``--convert-std-to-llvm`` pass. These passes are available in mlir-opt that should be located in ``llvm/build/bin``:
::

   $ mlir-opt --convert-scf-to-std --convert-std-to-llvm myFirst.mlir &> myFirst.llvm 

Execute *LLVM dialect* on CPU using ``mlir-cpu-runner`` that is located in ``llvm/build/bin``. The shared libraries (``libcomet_runner_utils`` and ``libmlir_runner_utils``) are utilized for runtime calls.
::
   
   $ mlir-cpu-runner myFirst.llvm -O3 -e main -entry-point-result=void -shared-libs=build/lib/libcomet_runner_utils.dylib,llvm/build/lib/libmlir_runner_utils.dylib

.. autosummary::
   :toctree: generated

