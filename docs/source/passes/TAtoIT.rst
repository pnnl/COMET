``convert-ta-to-it``
====================

The ``convert-ta-to-it`` pass converts the tensor algebra dialect to the index tree dialect.
This pass is mainly for lowering of multiplication and element-wise operations.
To perform efficient code generation for sparse tensors, it is recommended to use this pass in conjunction with ``opt-workspace`` pass to enable workspace transformations.

.. autosummary::
   :toctree: generated

