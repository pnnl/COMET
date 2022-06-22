COMET DSL
==========

The goal of the COMET language is to allow domain scientists to:

#. express concepts and operations in a form that closely resembles familiar notations, and,
#. convey domain-specific information to the compiler for better program optimization.

COMET uses Einstein mathematical notation and provides users with an interface to express tensor algebra semantics.
An example program in COMET DSL to perform dense GEMM is shown below.
A tensor object (e.g., ``A``) refers to a multi-dimensional array of arithmetic values that can be accessed by using indexing values.
Range-based index label constructs (e.g., [*i, j*])
represent the range of indices expressed through a scalar, a range, or a range with increment.
Index labels can be used both for constructing a tensor or for representing a tensor operation.
In a tensor construction, index labels are used to represent each dimension size.
In the context of a tensor operation, they represent slicing information of the tensor object where the operation will be applied.

The property of the tensor such as dense or sparse is also captured during the tensor declaration.
The COMET DSL allows users to express tensor’s properties for each dimension.
This information is later used to perform various optimizations during code generation, especially for sparse tensors.
There are utility runtime functions inside COMET that allow populating tensors. 
For example, ``random()`` initializes all the elements of a dense tensor with random values.

The various tensor operations supported inside COMET are listed in the :doc:`../operations` section.
In the program below, a matrix multiplication operation is performed between two matrices and the output is stored in a new dense matrix.
COMET recognizes that the following is multiplication operation between two dense matrices and generates code accordingly. 
Similarly, COMET can recognize mixed mode and pure sparse computation cases, and generate efficient code. 

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
     A[i, j] = random();                   # random initialization
     B[j, k] = random();
     C[i, k] = 0.0;                        # all values are initialized to 0

     # Tensor Contraction
     C[i, k] = A[i, j] * B[j, k];          # perform matrix multiplication

     print(C);                             # print the matrix
   }

.. autosummary::
   :toctree: generated

