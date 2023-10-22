Element-wise Multiplication
===========================

The element-wise binary operator ``.*`` in COMET computes a new tensor with elements that are obtained by multiplying corresponding elements of the two input tensors.
This operator works with both sparse and dense tensors.
Optimizations inside COMET are applied according to the type of input.
For example, when one input is sparse and the other is dense, and the user expects a sparse output, only effectual computations are performed.
A single line statement to perform element-wise multiplication between two 4 dimensional tensors is as follows:
::

  C[a, b, c, d] = A[a, b, c, d] .* B[a, b, c, d]     # COMET DSL

  C = comet.multiply(A, B)                           # Python NumPy
  
  C = A .* B;                                         # Rust eDSL

The following is an example of element-wise multiplication of a sparse matrix and dense matrix in COMET DSL:
::

   def main() {
     # IndexLabel declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];

     # Tensor declarations
     Tensor<double> A([i, j], {COO});      # declare a sparse tensor in COO format
     Tensor<double> B([i, j], {Dense});    # declare a dense tensor
     Tensor<double> C([i, j], {COO});

     # Tensor Fill operation
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0
     B[i, j] = 3.0;                        # initialize the dense matrix with all 3.0s

     C[i, j] = A[i, j] .* B[i, j];          # perform element-wise multiplication

     print(C);                             # print the sparse output matrix
   }

The following is an example of element-wise multiplication of a sparse matrix and sparse matrix in COMET DSL:
::

   def main() {
     # IndexLabel declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];

     # Tensor declarations
     Tensor<double> A([i, j], {CSR});      # declare a sparse tensor in CSR format
     Tensor<double> B([i, j], {CSR});
     Tensor<double> C([i, j], {CSR});

     # Tensor Fill operation
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0
     B[i, j] = comet_read(1);          # read in a sparse matrix @SPARSE_FILE_NAME1

     C[i, j] = A[i, j] .* B[i, j];         # perform element-wise multiplication

     print(C);                             # print the sparse output matrix
   }

.. autosummary::
   :toctree: generated

