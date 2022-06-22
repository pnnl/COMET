Workspace Transformations
=========================

The workspace transformation is applied on the index tree representation and useful for efficient sparse computations.
Temporary intermediate dense data structures, known as, workspace, are used to represent sparse computations.
The accesses to dense workspace data structures avoids inefficient accesses to sparse data structures.

Workspace transformations on the index tree allow generation of sparse output.
Storing an output tensor in dense formation is not optimal due to its large storage overhead that may cause runtime memory errors,
and also that the sparse output may be used for subsequent operations requiring an expensive translation step.

.. autosummary::
   :toctree: generated

