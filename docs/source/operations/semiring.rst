Semiring
========

The semiring operation ``@(x, y)`` is composed of two binary operations ``x`` and ``y``.
An easy way to understand semiring is think along the lines of matrix multiplication, where multiplication is performed first and then sum operation is performed.
The semiring replaces the multiplication and sum operations in matrix multiplication by ``y`` and ``x`` operations, respectively.
`GraphBLAS <https://people.engr.tamu.edu/davis/GraphBLAS.html>`_ shows that many graph algorithms can be expressed as linear algebra operations over semirings.
For example, all pairs shortest path problem can be formulated as matrix multiplication over min-plus semiring.
The supported semiring operations in COMET are listed in the table below:

.. csv-table:: Semiring Operations in COMET
   :header: "Semiring", "Operations", "Description"
   :widths: 12, 10, 20

   "Min-first", "@(min, first)", "‘min’ means the minimal value; ‘first’ means first(x, y) = x: output the value of the first in the pair."
   "Plus-mul", "@(+, * )", "‘+’ means addition; ‘*’ means multiplication."
   "Any-Pair", "@(any, pair)", "‘any’ means “if there is any; if yes return true”. ‘pair’ means pair(x, y) = 1: x and y both have defined value at this intersection."
   "Min-Plus", "@(min, +)", "‘min’ means the minimal value; ‘+’ means addition."
   "Plus-Pair", "@(+, pair)", "‘+’ means addition; ‘pair’ means pair(x, y) = 1: x and y both have defined value at this intersection."
   "Min-Second", "@(min, second)", "‘min’ means the minimal value; ‘second’ means second(x, y) = x: output the value of the second in the pair."
   "Plus-Second", "@(+, second)", "‘+’ means addition; ‘second’ means second(x, y) = x: output the value of the second in the pair."
   "Plus-first", "@(+, first)", "‘+’ means addition; ‘first’ means first(x, y) = x: output the value of the first in the pair."



The following is an example of semiring operation in COMET DSL:
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

     C[i, k] = A[i, j] @(+,*) B[j, k];     # perform matrix multiplication according to the provided semiring

     print(C);                             # print the sparse matrix in CSR format
   }

The following is an example of monoid operation in COMET DSL:
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

     C[i, j] = A[i, j] @(+) B[i, j];       # perform element-wise addition using monoid operator


     print(C);                             # print the sparse output matrix
   }

The following is an example of semiring operation in Rust eDSL:

.. highlight:: rust

::

   comet_rs::comet_fn! { mm_semiring_plustimes_csr_csr_csr, {
      let a = Index::new();                                              // dynamic index label, evaluated after file read
      let b = Index::new();
      let c = Index::new();

      let A = Tensor::<f64>::csr([a, b]).fill_from_file("path/to/file"); // declare a sparse tensor in CSR format
      let B = Tensor::<f64>::csr([b, c]).fill_from_file("path/to/file"); // and read in a sparse matrix 
      let C = Tensor::<f64>::csr([a, c]);
      C = A @(+,*) B;                                                    // perform matrix multiplication according to the provided semiring
      C.print();                                                         // print the sparse matrix in CSR format
   }}

   fn main() {
      mm_semiring_plustimes_csr_csr_csr();                               // call the function
   }



 
.. autosummary::
   :toctree: generated

