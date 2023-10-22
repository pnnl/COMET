FAQs
====

Here, we will maintain a list of FAQs.

#. *What compressed storage formats are supported to represent sparse tensors?*
   Sparse tensors can be represented using CSR (2D tensors) or CSF (> 2D tensors). 
   Internally, COMET uses a standard representation (see `reference <https://arxiv.org/pdf/2102.05187.pdf>`_) that can express most compressed storage formats like CSR, DCSR, or COO.
   
#. *How does COMET populate sparse tensors?*
   COMET DSL supports reading of sparse matrices from .mtx (`matrix market format <https://math.nist.gov/MatrixMarket/formats.html>`_) files.
   Whereas, .tns (`FROSTT file format <http://frostt.io/tensors/file-formats.html>`_) files are used for populating sparse tensors.
   The .mtx and .tns files are human readable text files where each line represents a non-zero element. 
   The runtime function gets an integer input (``comet_read(0)``) that is correlated with the user-defined environment variable ``SPARSE_FILE_NAME0`` appended with integer input provided as argument to the runtime function.

#. *Where can one find examples of sparse matrices and tensors?*
   The `SuiteSparse Matrix Collection <https://sparse.tamu.edu/>`_ has an ample collection of sparse matrices.
   The Formidable Repository of Open Sparse Tensors and Tools (`FROSTT <http://frostt.io/tensors/>`_) contains some higher order tensors. 
   

.. autosummary::
   :toctree: generated

