Transpose-Transpose-GEMM-Transpose
==================================

Tensor contractions are high-dimension analogs of matrix multiplications widely used in many scientific and engineering domains,
including deep learning, quantum chemistry, and finite-element methods.
Tensor contractions are computationally intensive and dominate the execution time of many computational applications.
These operations can be reformulated by transposing multi-dimensional input tensors into 2D matrices, performing a General Matrix Multiply (GEMM) operation, 
and unflatting the output tensor back to its original form as detailed below.
Although, this approach incurs the additional overhead of transpose operations, employing highly-optimized GEMM kernels outweighs this overhead. 

Consider the following tensor contraction, expressed using Einstein notation,
where two 4D tensors, ``A`` and ``B``, are contracted to produce a 4D tensor:

.. math::

    C[a, b, c, d] = A[a, e, b, f] * B[d, f, c, e]

In this contraction, the indices *e* appear in both right-hand tensors but not in the left-hand tensor ``C`` (summation or contraction
indices). The indices *a, b, c, d* appear in exactly one of the two input tensors and the output tensor (external or free indices).
A tensor contraction is, thus, the contraction of the two input tensors ``A`` and ``B`` over the contraction indices *e, f*:

.. math::

   C[a, b, c, d] = \sum_{e,f} A[a, e, b, f] * B[d, f, c, e]

A naïve nested-loop implementation of the above computation is inefficient due to poor data locality.
A more efficient approach, commonly used in modern high-performance tensor libraries, leverages highly optimized GEMM engines.
This approach, often referred as transpose-transpose-GEMM-transpose (TTGT), performs the permutations of the input tensors
followed by a high-performance matrix-matrix multiplication and a final permutation to reconstruct the output tensor. 
The first two transposes “flatten” a multi-dimensional tensor into a 2D matrix by first permutating the indices so that they are contiguous in memory 
and then merging pairs of consecutive indices to form lower-dimensional tensors as follows:

.. math::

   A[a, e, b, f] \rightarrow TA[a, b, e, f] = A_p[i, j] \\ 
   B[d, f, c, e] \rightarrow TB[e, f, d, c] = B_p[j, k]

The 4D tensor contraction can then be expressed as:

.. math::
 
   C_p[i, k] = A_p[i, j] * B_p[j, k]

Here, 

.. math::

   (a, b) \rightarrow i, (e, f) \rightarrow j, (d, c) \rightarrow k,\\
   C_p[i, k] = TC[a, b, d, c] \rightarrow C[a, b, c, d].

The TTGT method is effective to perform high-efficient tensor contractions despite the overhead of performing three additional permutations.
In fact, highly-optimized GEMM operations perform considerably better than nested loop implementations on modern architectures and exploit high data locality.

.. autosummary::
   :toctree: generated

