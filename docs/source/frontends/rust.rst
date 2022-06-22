Rust eDSL
=========

The COMET compiler supports the Rust eDSL front-end. 
Through the use of Rust procedural macros, the comet-rs crate (available at `crates.io <https://crates.io/crates/comet-rs>`_) exposes to the user a Rust-inspired DSL for implementing and executing COMET programs.
At compile time (of the Rust application) the ``comet_fn!`` macro parses the eDSL code, and invokes the COMET compiler to lower the eDSL code to various IR dialects and then compiles the code
into a shared library that is dynamically loaded at runtime.
These COMET compiled functions are executed at runtime by calling them as you would any other function.

*Requirements:*
Currently we have only tested the Rust Frontend on Linux.

* Rust - https://www.rust-lang.org/tools/install
  
  * we have tested with Stable Rust 1.60.0

* Build and test COMET 
  
  * please refer to the :doc:`../started/install` section for the build instructions

*Usage:*
Add the following to your project `Cargo.toml` file:

.. highlight:: rust

::

   [dependencies]
   comet-rs = "0.1.0"

If you have followed the COMET build instructions exactly, you will only need to set the following envrionment variable:

::

   COMET_DIR=/path/to/COMET/root/dir

If you have changed the build locations from what was listed in the COMET build instructions you will instead need to set the following environment variables:

::
   
   COMET_BIN_DIR=/path/to/COMET/bin/dir
   COMET_LIB_DIR=/path/to/COMET/lib/dir
   MLIR_BIN_DIR=/path/to/MLIR/bin/dir
   MLIR_LIB_DIR=/path/to/MLIR/lib/dir

Note that the ``MLIR_BIN_DIR`` and ``MLIR_LIB_DIR`` must point to the MLIR binaries and libraries built as part of the COMET build process. External LLVM builds are not supported.
You can now make use of the ``comet_fn!`` macro to define COMET functions in your Rust application.

.. highlight:: rust

::

   comet_rs::comet_fn! { print_dense, {
      let i = Index::with_value(4);
      let j = Index::with_value(4);

      let A = Tensor::<f64>::dense([i, j]).fill(2.3);

      A.print();
   }}

   comet_rs::comet_fn! { sum_coo, {
      let i = Index::new();
      let j = Index::new();

      let A = Tensor::<f64>::coo([i, j]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
      let a = A.sum();
      a.print();
   }}

   comet_rs::comet_fn! { dense_mm, {
      let i = Index::with_value(8);
      let j: Index = Index::with_value(4);
      let k = Index::with_value(2);
      let A = Tensor::<f64>::dense([i, j]).fill(2.2);
      let mut B = Tensor::<f64>::dense([j, k]).fill(3.4);
      let mut C = Tensor::<f64>::dense([i, k]).fill(0.0);
      C = A * B;
      C.print();
   }}

   fn main() {
      print_dense();
      sum_coo();
      dense_mm();
   }


.. autosummary::
   :toctree: generated

