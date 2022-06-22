COMET Optimizations
===================

The main advantage of a multi-level IR is that different kinds of optimizations can be applied at each level of the IR stack and can be shared across different stacks. 
In the COMET compiler, we apply domain-specific optimizations at the TA dialect and IT dialect level, general optimizations at the *linalg* dialect level,
and architecture-specific optimizations at the lower levels.

Some optimizations applied in COMET directly benefit some domains, such as those for tensor contractions has use in the computational chemistry domain.
Whereas, other optimizations have more broad applicability such as the workspace transforms that enable generation of sparse ouput benefits graph algorithms.
All optimizations performed inside the COMET compiler enable generation of efficient code.

As discussed in detail in the :doc:`../optimizations/workspace` section, the workspace transformations performed whilst the code is represented in the index tree dialect
facilitate generation of sparse tensor output.
Another optimization that improves the performance of tensor contractions is their reformulation to transpose-transpose-GEMM-transpose.
The multi-dimensional input tensors can be transposed into 2D matrices, next GEMM operation is performed, and the output tensor is un-flattened to its original form.
The use of highly-optimized GEMM kernels outweighs the overhead incurred by the transpose operations.
COMET also introduces runtime functions to implements some tasks efficiently, such as tensor transpose.
In some cases, COMET also applies reordering to matrices and tensors to optimize their memory access patterns.
For more details of COMET optimizations, see the :doc:`../optimizations` section.
 
.. autosummary::
   :toctree: generated

