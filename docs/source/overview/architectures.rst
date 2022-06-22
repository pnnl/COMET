Supported Architectures
=======================

In addition to supporting serial and parallel execution on CPUs using the LLVM IR as backend, COMET provides workflows (dialect conversion and code generation steps) 
that enable execution on GPUs, FPGAs, and spatial accelerators such as 
`Xilinx Versal AI Engine <https://www.xilinx.com/products/silicon-devices/acap/versal-ai-core.html>`_ and the `SambaNova system <https://sambanova.ai/>`_.
The *gpu* dialect inside MLIR is used to represent kernels, perform kernel optimizations, and convert code to PTX (for execution on NVIDIA GPUs) or vendor-independent SPIR-V dialect.
The MLIR code base supports only SPRIR-V shaders for execution with the Vulkan runtime, mainly targeting graphic applications. 
COMET adds support for SPIR-V kernel, targeting computational kernel execution supported by the OpenCL runtime. 
SPIR-V (kernel) representations can be used for execution on GPUs, as well as on FPGAs. 
At this time, we are testing the ability to generate kernel code that can be synthesized using vendor-specific high-level syntheis frameworks starting from SPIR-V binary IRs generated from COMET.
This workflow will be available in a future release.

Alternatively, Verilog code for FPGA execution can be generated through the `CIRCT project <https://circt.llvm.org/>`_, an active effort is being made to covert FIRRTL to Verilog HDL.
In this case, *std* dialect IR is lowered the to handshake dialect, which is then converted to  `FIRRTL <https://www.chisel-lang.org/firrtl/>`_ through the CIRCT infrastructure.
At this time, however, CIRCT has not fully matured to support generic computation on FPGA, thus this lowering path is still in its infancy and only relatively simple computation can be lowered.

The dataflow-like computing on spatial accelerators such as the Xilinx Versal AI Engine and SambaNova system is supported through translation of the SCF dialect into proprietary lower-level dialects that are specific for each vendor.
In both case, COMET produces a vendor-specific computational graph that can be downloaded to the dataflow accelerators for efficient execution.
While COMET is an open-source project, the vendors' backends may require specific licenses, hence we will not be able to release any code that interface with proprietary dialects and tools.
  
COMET leverages runtime for heterogeneous system to actually executes code on heterogeneous devices. While for CPU and GPU we can simply rely on the LLVM runtime or the CUDA/ROCm/Vulkan runtime, respectively, for execution on emerging architectures (as well as the ones previously mentioned), COMET will leveratge the  `MCL <https://github.com/pnnl/mcl>`_ runtime system for heterogeneous systems to schedule task scheduling on heterogeneous devices and perform resource management.
We expect to release this workflow in a future release. 

.. csv-table:: Status of support for various backend architectures inside COMET (release 0.1)
   :header: "Architecture", "Status", "Description"
   :widths: 10, 10, 15

   "GPU", "Active", "Support is provided through MLIR."
   "FPGA reconfigurable logic (Xilinx)", "Work-In-Progress", "Support for generating synthesizable kernels from COMET will be through SPIR-V translation."
   "FPGA Versal AI Engines (Xilinx)", "Vendor Proprietary", "Support is provided through lowering of SCF dialect to Xilinx proprietary dialect and binary using their compiler."
   "SambaNova", "Vendor Proprietary", "Support is provided through lowering of SCF dialect to SambaNova proprietary dialect and binary using their compiler."
   
.. note::
   
   The support for some architectures is a work-in-progress, therefore this page may be updated with future releases. 

.. autosummary::
   :toctree: generated

