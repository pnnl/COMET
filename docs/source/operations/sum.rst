Reduction
=========

The ``SUM`` unary operator in COMET returns the sum of all elements in the input tensor as a scalar element.
This operator works for both dense and any sparse formats, and can be expressed as follows: 
::

   var a = SUM(A[i, j])     # COMET DSL

   let a = A.sum();          # Rust eDSL

For dense tensors, nested for-loops are generated based on the size of tensor modes.
In particular, the number of for-loops is equal to the number of tensor modes, where the range of for-loops is the size of each tensor mode.
Then, values from each tensor element are loaded using for-loop inductions. 
The result of adding these values across iterations is stored in the output variable.
For sparse tensors of any sparse formats, a single for-loop is used to sum up only the non-zero values across iterations and the overall sum is stored in the output variable.

The following is an example of applying SUM to COO matrices in COMET DSL:
::

   def main() {
     # IndexLabel Declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];           

     # Tensor Declarations
     Tensor<double> A([i, j], {COO});      # declare a sparse tensor in COO format

     # Tensor Fill Operation 
     A[i, j] = comet_read(0);          # read in a sparse matrix @SPARSE_FILE_NAME0

     # Reduction operation
     var a = SUM(A[i, j]);                 # perform the reduction operation
     
     print(a);                             # print the output variable
   }

The following is an example of applying SUM to CSF tensors in COMET DSL:
::

   def main() {
     # IndexLabel Declarations
     IndexLabel [i] = [?];                 # dynamic index label, evaluated after file read
     IndexLabel [j] = [?];           
     IndexLabel [k] = [?];           

     # Tensor Declarations
     Tensor<double> A([i, j, k], {CSF});   # declare a sparse tensor in CSF format

     # Tensor Fill Operation 
     A[i, j, k] = comet_read(0);       # read in a sparse matrix @SPARSE_FILE_NAME0

     # Reduction Operation
     var a = SUM(A[i, j, k]);              # perform the reduction operation for 3D tensor
     
     print(a);                             # print the output variable
   }

.. autosummary::
   :toctree: generated

