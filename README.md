
# ⚡️ "COMET" / Domain specific COMpiler for Extreme Targets

The COMET compiler consists of a Domain Specific Language (DSL) for sparse and dense tensor algebra computations, a progressive lowering process to map high-level operations to low-level architectural resources, a series of optimizations performed in the lowering process, and various IR dialects to represent key concepts, operations, and types at each level of the multi-level IR. At each level of the IR stack, COMET performs different optimizations and code transformations. Domain-specific, hardware- agnostic optimizations that rely on high-level semantic information are applied at high-level IRs. These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices (e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).

## Documentation

Comprehensive documentation of the COMET compiler can be found [here](https://pnnl-comet.readthedocs.io/).

## Setting up COMET

These commands can be used to setup COMET project:

1) **Install Dependencies** cmake and ninja are required to install the COMET compiler.

2) **Check out LLVM and COMET repos.**  COMET contains LLVM as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support COMET.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

```
$ git clone https://github.com/pnnl/COMET.git
$ cd comet
$ git submodule init
$ git submodule update
```

*Note:* The repository is set up so that `git submodule update` performs a 
shallow clone, meaning it downloads just enough of the LLVM repository to check 
out the currently specified commit. If you wish to work with the full history of
the LLVM repository, you can manually "unshallow" the the submodule:

```
$ cd llvm
$ git fetch --unshallow
```

3) **Build and test LLVM/MLIR:**

```
$ cd comet
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-mlir
```

4) **Build and test COMET:**

```
$ cd comet
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-comet-integration # Run the integration tests.
```

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `Release` mode makes a very large difference
in performance.

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
* **Gokcen Kestor** (email: *first-name <dot> last-name <at> pnnl <dot> gov*), [Pacific Northwest National Laboratory (PNNL), United States.](https://www.pnnl.gov/)
* **Rizwan Ashraf** (email: *first-name <dot> last-name <at> pnnl <dot> gov*), [Pacific Northwest National Laboratory, United States.](https://www.pnnl.gov/)
* **Ryan Friese** (email: *first-name <dot> last-name <at> pnnl <dot> gov*), [Pacific Northwest National Laboratory, United States.](https://www.pnnl.gov/)

## Cite Our Project

If you use COMET in your research or work, please cite any of the following relevant papers:

* Erdal Mutlu, Ruiqin Tian, Bin Ren, Sriram Krishnamoorthy, Roberto Gioiosa, Jacques Pienaar & Gokcen Kestor, *COMET: A Domain-Specific Compilation of High-Performance Computational Chemistry,* In: Chapman, B., Moreira, J. (eds) Languages and Compilers for Parallel Computing, LCPC 2020, Lecture Notes in Computer Science, vol 13149, Springer, Cham. [DOI](https://doi.org/10.1007/978-3-030-95953-1_7) and [BIB](https://citation-needed.springer.com/v2/references/10.1007/978-3-030-95953-1_7?format=bibtex&flavour=citation).

::

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

* Ruiqin Tian, Luanzheng Guo, Jiajia Li, Bin Ren, & Gokcen Kestor, *A High Performance Sparse Tensor Algebra Compiler in MLIR,* In: IEEE/ACM 7th Workshop on the LLVM Compiler Infrastructure in HPC, LLVM-HPC 2021, November 14, 2021, St. Louis, MO, United States. [DOI](https://doi.org/10.1109/LLVMHPC54804.2021.00009) 

::

   @InProceedings{COMET:LLVM-HPC-2021,
      author={Tian, Ruiqin and Guo, Luanzheng and Li, Jiajia and Ren, Bin and Kestor, Gokcen},
      booktitle={2021 IEEE/ACM 7th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC)}, 
      title={A High Performance Sparse Tensor Algebra Compiler in MLIR}, 
      year={2021},
      pages={27-38},
      doi={10.1109/LLVMHPC54804.2021.00009}
   }
