
# ⚡️ "COMET" / Domain specific COMpiler for Extreme Targets

The COMET compiler consists of a Domain Specific Language (DSL) for sparse and dense tensor algebra computations, a progressive lowering process to map high-level operations to low-level architectural resources, a series of optimizations performed in the lowering process, and various IR dialects to represent key concepts, operations, and types at each level of the multi-level IR. At each level of the IR stack, COMET performs different optimizations and code transformations. Domain-specific, hardware- agnostic optimizations that rely on high-level semantic information are applied at high-level IRs. These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices (e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).

## Documentation

Comprehensive documentation of the COMET compiler can be found [here](https://pnnl-comet.readthedocs.io/).

## Setting up COMET

These commands can be used to setup COMET project:

1) **Requirements.** 
To install COMET and LLVM/MLIR, the following dependencies need to be already installed:
* [CMake (3.25 or later)](https://cmake.org/download)
* [Ninja (1.5 or later)](https://ninja-build.org/)
* C++ compiler toolchain as [mentioned here](https://llvm.org/docs/GettingStarted.html#requirements)
* [Python3 (3.9 or later)](https://www.python.org/downloads/)
* [Git (1.8.4 or later)](https://www.git-scm.com/)
* [pkg-config (0.29.2 or later)](https://www.freedesktop.org/wiki/Software/pkg-config/)

When targeting GPUs or/and FPGAs you will also need the drivers and runtimes of the respective vendors (Nvidia/CUDA, AMD/ROCm, Xilinx/XRT,Vitis).

   1.a **[Optional but recommended] Create a new python environment**
   ```bash
   $ export PYTHON_EXECUTABLE=$(which python3.x) # Replace 3.x with your version
   $ ${PYTHON_EXECUTABLE} -m venv "comet"
   $ source comet/bin/activate
   ```

2) **Build COMET.**  
LLVM and blis are dependencies included in this repo as git submodules that point to the respective versions of the libraries that COMET has been tested with. LLVM/MLIR are changing relatively rapidly, so feel free to use the current version of LLVM, but APIs may have changed. 

BLIS is an award-winning portable software framework for instantiating high-performance 
BLAS-like dense linear algebra libraries. COMET generates a call to BLIS microkernel 
after some optimizations. Also, blis is patched with changes specific to COMET, so an existing installation may not be used. 

To build COMET for CPU execution only, run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ../
$ make
```

This will fetch and build the LLVM and blis dependencies automatically, and build COMET. Once the command completes COMET will be installed in `build/comet/`. 
You may also specify a custom LLVM installation, instead of downloading a fresh copy, by passing its path to the `cmake` command:
```bash
$ cmake ../ -DLLVM_CUSTOM_BUILD_PATH=/path/to/llvm/build/
```

*Note*: The LLVM installation should have enabled the `mlir, openmp, clang` projects.
Once complete, you can run the integration tests using the following commands:
```bash
$ cd comet
$ ninja check-comet-integration
```

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into COMET.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `Release` mode makes a very large difference
in performance.

3) **Enabling GPU Support.**
To enable support for Nvidia, AMD GPUs you need to set the respective option in `cmake`:
```bash
# NVIDIA GPU
$ cmake ../ -DENABLE_NVIDIA_GPU_BACKEND=ON
$ make

# AMD GPU
$ cmake ../ -DENABLE_AMD_GPU_BACKEND=ON
$ make
```

This will download and install [Triton](https://github.com/triton-lang/triton), a MLIR dialect for targeting GPUs used by COMET as a backend, and enable the GPU-related options in `comet-opt`.  
You can also specify a default target device capability by passing the option
`-DDEVICE_COMPUTE_CAPABILITY=<gpu-capability>` in cmake. You can specify the same attribute later at the `comet-opt` command using the flag `--gpu-compute-capability`
Example options include `sm_80, sm_90` for Nvidia and `gfx908` for AMD.  For example:
```bash
# Example for NVIDIA GPU
$ cmake ../ -DENABLE_NVIDIA_GPU_BACKEND=ON -DDEVICE_COMPUTE_CAPABILITY=sm_90
$ make

# Example for AMD GPU
$ cmake ../ -DENABLE_AMD_GPU_BACKEND=ON -DDEVICE_COMPUTE_CAPABILITY=gfx908
$ make
```

4) **Enabling FPGA Support.**
To enable support for FPGAs, (currently only Xilinx/AMD), you need to set the flag `-DENABLE_FPGA_TARGET=ON` in `cmake`. The FPGA support relies on other dependencies including an older version of LLVM found as a submodule in `tools/llvm-spirv` and a LLVM-SPIRV translator found in `tools/spriv-llvm-translate`. Setting the above flag will automatically download and install these dependencies, as well as [MCL](https://minos-computing.github.io/) the runtime system used to issue interact with the FPGA. For more information see [here](tools/README.md).

## License

This project is licensed under the Simplified BSD License. 
See the [LICENSE file](https://github.com/pnnl/COMET/blob/master/LICENSE)
and the [DISCLAIMER file](https://github.com/pnnl/COMET/blob/master/DISCLAIMER.txt) for more details.

## Reporting Issues

Issues with COMET can be reported through [GitHub](https://github.com/pnnl/COMET/issues). 
We will try our best to timely address issues reported by users. 
The community is also welcome to discuss any remedies or experience that may help to resolve issues.

## Contributions

Contributions to COMET are welcome. The community can get involved by contributing some new feature, reporting bugs, and/or improving documentation. 
Please feel free to create a pull-request on [GitHub](https://github.com/pnnl/COMET/pulls) for code contributions. 
We will try our best to timely incorporate user requests.

## Contact Us

We encourage you to use GitHub’s tracking system to report any issues or for code contributions as mentioned above. 
For any other queries, please feel free to contact us via email:
* **Gokcen Kestor** (email: *first-name.last-name@pnnl.gov*), [Pacific Northwest National Laboratory (PNNL), United States.](https://www.pnnl.gov/)
* **Zhen Peng** (email: *first-name.last-name@pnnl.gov*), [Pacific Northwest National Laboratory, United States.](https://www.pnnl.gov/)
* **Polykarpos Thomadakis** (email: *first-name.last-name@pnnl.gov*), [Pacific Northwest National Laboratory, United States.](https://www.pnnl.gov/)
* **Ryan Friese** (email: *first-name.last-name@pnnl.gov*), [Pacific Northwest National Laboratory, United States.](https://www.pnnl.gov/)

## Cite Our Project

If you use COMET in your research or work, please cite any of the following relevant papers:

* Erdal Mutlu, Ruiqin Tian, Bin Ren, Sriram Krishnamoorthy, Roberto Gioiosa, Jacques Pienaar & Gokcen Kestor, *COMET: A Domain-Specific Compilation of High-Performance Computational Chemistry,* In: Chapman, B., Moreira, J. (eds) Languages and Compilers for Parallel Computing, LCPC 2020, Lecture Notes in Computer Science, vol 13149, Springer, Cham. [DOI](https://doi.org/10.1007/978-3-030-95953-1_7) and [BIB](https://citation-needed.springer.com/v2/references/10.1007/978-3-030-95953-1_7?format=bibtex&flavour=citation).
```
   @InProceedings{COMET:LCPC-20,
      author={Mutlu, Erdal and Tian, Ruiqin and Ren, Bin and Krishnamoorthy, Sriram and Gioiosa, Roberto and Pienaar, Jacques and Kestor, Gokcen",
      editor={Chapman, Barbara and Moreira, Jos{\'e}},
      title={COMET: A Domain-Specific Compilation of High-Performance Computational Chemistry},
      booktitle={Languages and Compilers for Parallel Computing},
      year={2022},
      publisher={Springer International Publishing},
      address={Cham},
      pages={87--103}
    }
```
* Ruiqin Tian, Luanzheng Guo, Jiajia Li, Bin Ren, & Gokcen Kestor, *A High Performance Sparse Tensor Algebra Compiler in MLIR,* In: IEEE/ACM 7th Workshop on the LLVM Compiler Infrastructure in HPC, LLVM-HPC 2021, November 14, 2021, St. Louis, MO, United States. [DOI](https://doi.org/10.1109/LLVMHPC54804.2021.00009) 
```
   @InProceedings{COMET:LLVM-HPC-2021,
      author={Tian, Ruiqin and Guo, Luanzheng and Li, Jiajia and Ren, Bin and Kestor, Gokcen},
      booktitle={2021 IEEE/ACM 7th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC)}, 
      title={A High Performance Sparse Tensor Algebra Compiler in MLIR}, 
      year={2021},
      pages={27-38},
      doi={10.1109/LLVMHPC54804.2021.00009}
   }
```

## Support

The COMET compiler is supported in part by the [Data-Model Convergence (DMC)](https://www.pnnl.gov/projects/dmc) 
initiative at the [Pacific Northwest National Laboratory](https://www.pnnl.gov/).

This work is also supported in part by the [High Performance Data Analytics (HPDA)](https://www.pnnl.gov/computing/HPDA/) program
at the [Pacific Northwest National Laboratory](https://www.pnnl.gov/).
 
This work is also supported in part by the U.S. Department of Energy’s (DOE) [Office of Advanced Scientific Computing Research (ASCR)](https://www.energy.gov/science/ascr/advanced-scientific-computing-research>)
as part of the [Center for Artificial Intelligence-focused Architectures and Algorithms (ARIAA)](https://www.pnnl.gov/projects/co-design-center-artificial-intelligence-focused-architectures-and-algorithms).

![alt text](https://github.com/pnnl/COMET/blob/master/docs/source/DMC_PNNL.jpeg)
