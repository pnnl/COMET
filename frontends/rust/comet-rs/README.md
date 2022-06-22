# Comet-rs
This crate is a Rust-based eDSL front end for the COMET compiler.


## ⚡️ "COMET" / Domain specific COMpiler for Extreme Targets

The COMET compiler consists of a Domain Specific Language (DSL) for sparse and dense tensor algebra computations, a progressive lowering process to map high-level operations to low-level architectural resources, a series of optimizations performed in the lowering process, and various IR dialects to represent key concepts, operations, and types at each level of the multi-level IR. At each level of the IR stack, COMET performs different optimizations and code transformations. Domain-specific, hardware- agnostic optimizations that rely on high-level semantic information are applied at high-level IRs. These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices (e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).

Through the use of procedural macros, this crate exposes to the user a Rust based eDSL which will be lowered to various dialects and then compiled into a shared library at compile time of the user application. At runtime the shared library is dynamically linked to, exposing the compiled COMET functions, allowing the user to execute them as the would any other function.

## External Dependencies
 - [COMET](https://github.com/pnnl/COMET)

Please follow the build instructions in the COMET repository to install the COMET compiler. (Also included in the COMET installation section below)

## Cargo.toml
 ```toml
 [dependencies]
 comet-rs = "0.1.0"
```
## Required Environment Variables
If you have followed the COMET build instructions exactly, you will only need to set the following envrionment variable:
```text
COMET_DIR=/path/to/COMET/root/dir
```

If you have changed the build locations from what was listed in the COMET build instructions you will instead need to set the following environment variables:
 ```text
 COMET_BIN_DIR=/path/to/COMET/bin/dir
 COMET_LIB_DIR=/path/to/COMET/lib/dir
 MLIR_BIN_DIR=/path/to/MLIR/bin/dir
 MLIR_LIB_DIR=/path/to/MLIR/lib/dir
 ```
 *Note that as part of the COMET build process we will also build a specific version of MLIR (managed as a git submodule),
 COMET will only work with this specific commit, so please do not point to a different MLIR version you might have build outside the COMET build process.*

## Example

 COMET uses Einstein mathematical notation and The `comet_fn!` macro provides users with an interface to express tensor algebra semantics using a Rust-like eDSL.

 ```rust
 use comet_rs::*;

 comet_fn! { dense_dense_matrix_multiply, {
    let a = Index::with_value(2);
    let b = Index::with_value(2);
    let c = Index::with_value(2);

    let A = Tensor::<f64>::dense([a, b]).fill(2.2);
    let B = Tensor::<f64>::dense([b, c]).fill(3.4);
    let C = Tensor::<f64>::dense([a, c]).fill(0.0);
    C = A * B;
    C.print();
 }}

 fn main() {
     dense_dense_matrix_multiply();
 }
 ```
## Operations
 
 We have implemented the following tensor operations (most of which are not valid rust syntax, but are valid COMET eDSL syntax)
 please refer to the [COMET documentation](https://pnnl-comet.readthedocs.io/en/latest/operations.html) for more in-depth descriptions of each operation.
 - Multiplication: `A * B`
 - Elementwise Multiplication: `A .* B`
 - Semiring Operations: `@(op1, op2)`
   - Min: ` A @(min) B`
   - Plus: `A @(+) B`
   - Mul: `A @(*) B`
   - Any-Pair: `A @(any,pair) B`
   - Plus-Mul: `A @(+,*)B `
   - Plus-Pair: `A @(+,pair) B`
   - Plus-First: `A @(+,first) B`
   - Plus-Second: `A @(+,Second) B`
   - Min-Plus: `A @(min,+) B` 
   - Min-First: `A @(min,first) B`
   - Min-Second: `A @(min,second) B`
 - Transpose: `B = A.transpose()`
 
## Optimizations
 We also support the ability to specify various optimizations to be performed by the COMET compiler.
 please refer to the [COMET documentation](https://pnnl-comet.readthedocs.io/en/latest/optimizations.html) for more in-depth descriptions of each optimization.
 - [Permutation TTGT](https://pnnl-comet.readthedocs.io/en/latest/passes/PermTTGT.html): `BestPermTtgt`
 - [Tensor Algebra to Index Tree](https://pnnl-comet.readthedocs.io/en/latest/passes/TAtoIT.html): `TaToIt`
 - [Tensor Contraction to TTGT](https://pnnl-comet.readthedocs.io/en/latest/passes/TC.html): `TcToTtgt`
 - [Loops](https://pnnl-comet.readthedocs.io/en/latest/passes/loops.html): `ToLoops`
 - [Matmult Kernel](https://pnnl-comet.readthedocs.io/en/latest/passes/mkernel.html): `MatMulKernel`
 - [Matmult Tiling](https://pnnl-comet.readthedocs.io/en/latest/passes/tiling.html): `MatMulTiling`
 - [Dense Transpose](https://pnnl-comet.readthedocs.io/en/latest/passes/transpose.html): `DenseTranspose`
 - [Workspace](https://pnnl-comet.readthedocs.io/en/latest/passes/workspace.html): `CompWorkspace`

 The above optimizations can be passed to the compiler as part of a custom syntax proivded as an argument to the `comet_fn` macro. 

### Example
 ```rust
 comet_fn! {function_name, {
         eDSL code
 },
 CometOptions::[TcToTtgt, BestPermTtgt, ToLoop]
 }
 ``` 

## COMET Output
 During evaluation of the `comet_fn` macro, the COMET compiler will generate a shared library containing the compiled COMET function.
 The shared library will be located in the same directory as the user application in a directory labelled `comet_libs`.
 This crate handles linking this shared library into your application automatically, but it depends on the library remaining in the `comet_libs` directory

 ## Crate Features
 By default if a COMET function fails to compile, it will also cause the overall Rust application to fail.
 It may be useful in some cases (e.g running the unit tests provided in this crate) to have a failed COMET compilation only emit a warning instead of an error.
 In this case the procedure macro simply creates rust code that will print out the COMET compiler error instead of executing the function.
 This functionality can be enabled by specifying the `comet_errors_as_warnings` feature:
 ```
 cargo test --no-fail-fast --features comet_errors_as_warnings
 ```
## COMET Installation Instructions

These commands can be used to setup COMET project:
This crate will not work without a successful COMET installation.
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
