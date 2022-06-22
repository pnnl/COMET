Multi-operand Expression
========================

In a multi-operand tensor expression, the order in which contractions are computed may effect the number of operations that need to be performed.
Given the associative property of tensor contractions, grouping order of the contractions may lead to significant performance advantage as long as the expression produces the correct result.
Performance variation may be significant, especially if some of the tensors involved have low cardinality in some dimensions (e.g., “skinny matrices”).
Therefore, all possible orderings of a multi-operand expression are exhaustively explored and the one that minimizes the overall number of operations is chosen.
Then, the sequence of operations are organized in a binary tree, and the multi-operand expression is lowered to a sequence of tensor contractions.
Note, that because the shape of the intermediate tensors is different from the original one, some tensor contractions may degenerate
to simpler lower-dimension operations, such as GEMM or tensor-vector multiplications, which are further optimized (e.g., removing additional transpose).

.. autosummary::
   :toctree: generated

