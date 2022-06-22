Python NumPy
=============

The COMET compiler supports the Python NumPy front-end. 
Some common NumPy methods are realized in COMET Python package. 
Just-in-time compilation is used and compilation flags can be passed to COMET using a Python decorator.

*Requirements*

* Python 3 dependencies required are as follows:

  * NumPy
  * ast
  * inspect
  * jinja2
* Install COMET Python package as follows:
  
  * *For macOS (>= Catalina) users:* pip install -i https://test.pypi.org/simple/ comet-pkg==2.2
  * *For Linux (Ubuntu) users:* pip install -i https://test.pypi.org/simple/ comet-lnx==1.1

*Usage*

To use, import comet in your python code : ``from comet_pkg import comet``.
The python decorator ``@comet.compile(flags=...)`` is used with method(s) to compile using COMET.
The "numpy" computations in a target method need to be replaced with keyword "comet," e.g. ``numpy.einsum()`` is replaced with ``comet.einsum()``.
The following is an example of dense GEMM performed using the COMET's NumPy front-end. 

::

   import numpy as np
   from comet_pkg import comet

   # Perform Matrix Multiplication
   @comet.compile (flags="--convert-ta-to-it --convert-to-loops")
   def compute_einsum_2D_comet (A, B):

      C = comet.einsum ('ij,jk->ik', A, B)

   return C

   A = np.full ((2,3),1, dtype=float)
   B = np.full ((3,4), 3, dtype=float)

   result = compute_einsum_2D_comet (A,B)
   print (result)

 
.. autosummary::
   :toctree: generated

