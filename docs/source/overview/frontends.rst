Supported Front-ends
====================

COMET Currently supports three front-ends:

* COMET DSL
* Python NumPy bindings
* Rust eDSL

COMET DSL
---------

COMET includes a high-level DSL that allows programmers to use Einstein notation to express tensor algebra semantics.
Computations on sparse and dense tensors is supported. 
The DSL also includes utility functions to read data from files, populate tensors, timing measurement, and print tensors. 
More details of realizing different operations in COMET DSL can be found in :doc:`../operations`.
A statement to perform tensor contraction on 4D tensors can be simply described as:
::

   C[a, b, c, d] = A[a, e, d, f] * B[b, f, c, e];      # 4D tensor contraction in COMET DSL

Here, ``A``, ``B`` and ``C``, are the names of the tensors, and *a*, *b*, *c*, *d*, *e* and *f* are the index labels used to define the tensors.


Python NumPy bindings
---------------------
Bindings have been developed to enable just-in-time compilation of supported methods described in Python language. 
As an example, the tensor contraction above can be expressed as below.
More details about the Python frontend can be found in :doc:`../frontends/python` section.
::

   C = comet.einsum ('aedf,bfce->abcd', A, B)          # 4D tensor contraction using NumPy-like interface.

Rust eDSL
---------
We have developed a Rust frontend that enables (Rust application) compile time compilation of supported methods using Rust procedural macros.
As an example, the tensor contraction above can be expressed as below.
More details about the Rust frontend can be found in :doc:`../frontends/rust` section.

.. highlight:: rust

::

   C = A * B // 4D tensor contraction using Rust eDSL 
             // index labels are captured and stored when the tensors are defined,
             // no need to explicity state them in the expression

.. note::
   Support for more front-ends may be added in the future based on the interest of the community. 

.. autosummary::
   :toctree: generated



