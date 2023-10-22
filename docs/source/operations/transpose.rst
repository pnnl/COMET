Transpose
=========

The ``transpose`` unary operation returns a new tensor that is transposed version of the input tensor.
If the dimensions of input tensor are *i* and *j*, then the output tensor dimensions will be *j* and *i*.
::

   B[j, i] = transpose(A[i, j], {j, i})          # COMET DSL

   B = A.transpose([j,i]);                       # Rust eDSL

The algorithm behind dense matrix/tensor transposition is that we swap the index order of the input matrix/tensor, 
and store it into the output matrix/tensor. 

For sparse matrix/tensor transposition, we implement it through a runtime function call. 
The function call takes the input matrix/tensor as input and outputs the transposed matrix/tensor, 
and currently, transpose of upto 3 dimensions is supported. 

The following is an example of Transpose to dense tensors in COMET DSL:
::

   def main() {
     # IndexLabel Declarations
     IndexLabel [i] = [4];                 # static index label
     IndexLabel [j] = [4];   
     IndexLabel [k] = [4];                     

     # Tensor Declarations
     Tensor<double> A([i, j, k], Dense);   # declare a dense tensor	  
     Tensor<double> B([k, i, j], Dense);

     # Tensor Random Initialization      
     A[i, j, k] = random();                # initialize the dense tensor with random values

     # Tensor Transpose
     B[k, i, j] = transpose (A[i, j, k], {k, i, j});
	
     print(B);                             # print the dense tensor
   }


The following is an example of Transposing CSR matrices in COMET DSL:
::

   def main() {
     # IndexLabel Declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];           

     # Tensor Declarations
     Tensor<double> A([i, j], CSR);	   # declare a sparse matrix in CSR format
     Tensor<double> B([j, i], CSR);

     # Tensor Readfile Operation      
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0

     # Tensor Transpose
     B[j, i] = transpose(A[i, j], {j, i}); # perform sparse transpose @SORT_TYPE
                                           # env. var SORT_TYPE selects the sorting algorithm

     print(B);                             # print the sparse output matrix
   }

.. autosummary::
   :toctree: generated

