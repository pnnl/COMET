Two-Phase Computation
=====================

One of the challenges of sparse computation comes from the unknown size and distribution of the output tensor. 
First, the number of nonzero elements in the output tensor is unknown before computation, which makes memory management very difficult. 
A general method is to allocate a very large chunk of memory to avoid the case where the output size exceeds the allocation, which results in redundant memory usage. 
Second, it is hard to know beforehand how nonzero elements are distributed among different rows (in case of row-major storage) in the output tensor. 
To update the output tensor in parallel, it is common to use a lock on the critical data structure, which results in high synchronization overhead.

To determine the needed size and real distribution of the output tensor, this work generates the code with two phases for sparse computation. 
The first phase is called the symbolic phase. It follows the same procedure of the given sparse computation (e.g., SpGEMM) in a "symbolic" way that it does not execute the computation, but only records the nonzero distribution of the output. 
After that, the symbolic phase can also determine the true number of nonzero elements in the output tensor and then allocate the sparse data structure with only the needed memory size. 
The second phase is called the numeric phase. It performs the real "numeric" computation with the prior knowledge from the symbolic phase. 
For some components of the sparse data structure (e.g., the index array in CSR), the output can be placed directly at the correct location that is provided by the symbolic phase. 
Therefore, the two-phase computation can minimize memory usage effectively and also enable parallelization for sparse computation.

.. autosummary::
   :toctree: generated

