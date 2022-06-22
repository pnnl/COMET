``convert-tc-to-ttgt``
======================

Tensor contractions can be reformulated by transposing multi-dimensional input tensors into 2D matrices, performing a GEMM operation, and unflatting the
output tensor back to its original form.
The pass to convert tensor contractions to TTGT can be employed using the ``convert-tc-to-ttgt`` flag.
See :doc:`../optimizations/ttgt` for more details of converting a tensor contraction to TTGT.

.. autosummary::
   :toctree: generated

