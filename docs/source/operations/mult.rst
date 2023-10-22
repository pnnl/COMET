Multiplication
===============

The multiplication operator in COMET DSL is ``*``. 
Generally, COMET employs index labels to determine the type of operation to perform. 
For example, the ``*`` operator refers to a tensor contraction if the contraction indices are adjacent.
Also, COMET automatically recognizes set of optimizations to perform based on the type of inputs, such as mixed mode (dense and sparse) or pure sparse computation.
A single line expression to perform tensor contraction between two 4 dimensional tensors is as follows:
::

    C[a, b, c, d] = A[a, e, d, f] * B[b, f, c, e];      # 4D tensor contraction in COMET DSL

    C = comet.einsum ('aedf,bfce->abcd', A, B)          # 4D tensor contraction using NumPy-like interface.

    C = A * B;                                           # 4D tensor contraction using Rust eDSL, no need to explicitly specify the indices.

The following is an example of SpMM (sparse matrix-dense matrix multiplication) in COMET DSL:
::

   def main() {
     # IndexLabel declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];
     IndexLabel [k] = [4];                 # static index label

     # Tensor declarations
     Tensor<double> A([i, j], {COO});      # declare a sparse tensor in COO format
     Tensor<double> B([j, k], {Dense});    # declare a dense tensor
     Tensor<double> C([i, k], {Dense});

     # Tensor Fill operation
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0
     B[j, k] = random();                   # initialize the dense matrix with random values
     C[i, k] = 0.0;                        # initialize the dense matrix with all 0s

     # Tensor Contraction
     C[i, k] = A[i, j] * B[j, k];          # perform spMM

     print(C);                             # print the dense matrix
   }

The following is an example of spGEMM (sparse matrix-sparse matrix multiplication) in COMET DSL:
::

   def main() {
     # IndexLabel declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];
     IndexLabel [k] = [?];

     # Tensor declarations
     Tensor<double> A([i, j], {CSR});      # declare a sparse tensor in CSR format
     Tensor<double> B([j, k], {CSR});
     Tensor<double> C([i, k], {CSR});

     # Tensor Fill operation
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0
     B[j, k] = comet_read(1);          # read in a sparse matrix @SPARSE_FILE_NAME1

     # Tensor Contraction
     C[i, k] = A[i, j] * B[j, k];          # perform spGEMM

     print(C);                             # print the sparse matrix in CSR format
   }
 
.. autosummary::
   :toctree: generated

