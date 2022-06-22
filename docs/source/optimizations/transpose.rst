Transpose
=========

Transpose is an important operation, especially with the use of TTGT method and therefore requires special consideration for optimal performance.
The optimization of dense and sparse tensors is treated separately.
For transpose of dense tensors, the optimization consists of two steps loop permutation and tiling.
Whereas, for transpose of sparse tensors, specialized runtime functions are implemented that utilize sorting.

In case of dense tensors, the main rationale of loop permutation is that the loops corresponding to the innermost indices should be at the innermost level.
A weight is assigned to each loop index according to its position in the input and output tensor 
(the weight is higher if an inner index does not correspond to an inner loop) and the overall cost of the permutation is computed by summing these weights.
Next, tiling is employed to improve locality.

In case of sparse tensors, COMET provides the option to use sorting and re-traversal of the input from transposed direction.
When using the sorting option, multiple sorting algorithms such as quick sort, count sort, bucket sort, radix sort, mixed count-bucket sort, and mixed count-radix-bucket sort are available.
In this case, the input tensor is converted to the coordinate (COO) format, the order of dimensions is then swapped to match the transposed order,
and sorting is performed on each dimension in ascending order.
The final step is the conversion to the target sparse format as desired by the user.

.. autosummary::
   :toctree: generated

