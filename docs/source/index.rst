Welcome to COMET's documentation!
===================================

The **COMET** compiler consists of a *Domain Specific Language* (DSL) for sparse and dense tensor algebra computations, 
a progressive lowering process to map high-level operations to low-level architectural resources, 
a series of optimizations performed in the lowering process, 
and various intermediate representation (IR) dialects to represent key concepts, operations, 
and types at each level of the multi-level IR. 

At each level of the IR stack, COMET performs different optimizations and code transformations. 
Domain-specific, hardware-agnostic optimizations that rely on high-level semantic information are applied at high-level IRs.
These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices 
(e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) 
and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).

Check out the :doc:`intro` section for further information, including
how to :doc:`install` and :doc:`test` COMET.

.. note::

   This project is under active development and this documentation is not complete.

Contents
--------

.. toctree::

   intro
   install
   test
