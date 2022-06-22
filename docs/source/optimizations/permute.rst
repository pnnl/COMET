TTGT Optimal Permutation
========================

The permutation chosen to reformulate a tensor contraction using the TTGT method has a considerable impact on performance.
The cost of transposing a high-dimension tensor into a 2D tensor depends on which indices are transposed and the storage format. 
For row-major format, transposing the first indices is more expensive than transposing the later ones, especially for the output tensor.
In order to select the best permutation, a heuristic cost model is used that assigns higher costs to permutations that move the outermost dimensions.
Additionally, some permutation naturally results in a reduction of the number of transpose operations.
The cost of each valid transposition of input and output tensors is computed, including the position swap for the input tensors, 
and the permutation with the lowest cost is selected.

.. autosummary::
   :toctree: generated

