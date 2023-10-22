COMET Dialects
==============

The COMET compiler introduces two dialects into the MLIR framework, namely, the Tensor Algebra (TA) and Index Tree (IT) dialects. 
The dialects are internal representations used by  COMET  to perform various optimizations.
The TA dialect supports mix dense/sparse tensor algebra computations with a wide range of storage formats. 
The IT dialect, on the other hand, is used for efficient code generation of sparse computations using an index tree representation for tensor expressions.
In most cases, users do not have to directly interact with these internal representations. They are used internally to carry semantic information to the later stages of the compiler and to generate highly efficient executables. 

Tensor Algebra Dialect
----------------------
The first dialect in the COMET compiler stack is the TA dialect. 
The main goal of this dialect is to represent basic building blocks of the tensor algebra computation, describe
tensor algebra specific types and operations, and represent semantics information expressed through the COMET DSL and other front-ends.

Most operators from the COMET DSL correspond to an operation in the TA dialect.
An ``index_label`` operation in TA dialect corresponds to an ``IndexLabel`` construct in the COMET DSL.
New tensors are declared with the ``tensor_decl`` operation, which takes as input the index labels for each dimension and the data type.
Three classes of tensor operations are currently supported: unary (fill, copy, set), binary (contraction), and multi-operand expressions (contraction chains).
The ``fill`` operation initializes a tensor object with a single value that is provided as an attribute.
The ``copy`` operation performs an element-wise copy between two tensors scaling the output tensor by factor *alpha*.
The ``set`` operation operates similarly to the ``copy`` operation but takes as input the result of a binary operation instead of a tensor. 
This operation is used to support multi-operand tensor contractions.
Tensor contractions ``tc`` take as input the input and output tensors, the scaling value *alpha*, and the indexing maps for the labels used in the contraction.

For multi-operand expressions that involve several contractions, a utility operation (``mult``) is introduced that represents a binary operation.
The actual computation for a multi-operand expression includes calculation of intermediates and then the actual tensor contractions. 
The order of binary operations is represented with a binary tree and the results are assigned to the output tensor using the ``set`` operation.

Some operations in TA dialect are lowered directly to dialects available in MLIR framework, while others are translated to the IT dialect inside COMET.
For example, if someone declares a dense tensor, these are lowered into ``alloc`` and ``tensor_load`` operations in *std* dialect. 
Similarly, when multiple arrays are allocated in case of declaration of a sparse tensor (position, coordinate, and non-zero value arrays).
The ``fill`` operator in the TA dialect initializes these data arrays using operations from the *linalg* dialect inside MLIR.
Whereas, when the tensor contraction expression is lowered into the ``tc`` operation in the TA dialect. 
The lowering routines takes the ``tc`` operation and generates operations from the IT dialect.

Index Tree Dialect
------------------
The IT dialect inside COMET is used to represent the loop order and the computation information for a tensor expression via an index tree. 
The index tree representation consists of two types of nodes: index nodes and compute nodes.
The index nodes contain a list of indices for nested for-loops to iterate on (includes the order). Compute nodes contain the compute statements of the tensor expression.
A sparse tensor contraction operation in TA dialect is converted to ``itree``, ``Indices`` and ``Compute`` operations corresponding to the root node, index node and compute node of the index tree, respectively.

The computational kernel code that is a combination of *scf* and *std* dialects is generated based on the types of inputs.
The IT dialect conversion starts from root of the index tree.
The code generation algorithm traverses the index tree in depth-first order to generate operations for each node, including inner tree node ``Indices`` and leaf node ``Compute``.
The ``Indices`` operations are converted to ``scf.for`` operation. Whereas, the ``Compute`` operations are converted into a set of standard dialect operations.
They will be put into the body of the generated for loops, which are generated from the parent operation of the ``Compute`` operation.

The conversion to loops in *scf* dialect enables optimizations to be applied using MLIR.
Here, low-level optimizations are applied to enable execution on target architecture.
The devices supported by the COMET compiler are listed in the :doc:`../overview/architectures` section.

.. autosummary::
   :toctree: generated

