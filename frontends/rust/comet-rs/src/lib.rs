//! Comet-rs is a Rust-based eDSL frontend for the COMET compiler.
//!
//! The COMET compiler consists of a Domain Specific Language (DSL) for sparse and dense tensor algebra computations,
//! a progressive lowering process to map high-level operations to low-level architectural resources, 
//! a series of optimizations performed in the lowering process, and various IR dialects to represent key concepts, operations, 
//! and types at each level of the multi-level IR. At each level of the IR stack, 
//! COMET performs different optimizations and code transformations. 
//! Domain-specific, hardware- agnostic optimizations that rely on high-level semantic information are applied at high-level IRs. 
//! These include reformulation of high-level operations in a form that is amenable for execution on heterogeneous devices 
//! (e.g., rewriting Tensor contraction operations as Transpose-Transpose-GEMM-Transpose) 
//! and automatic parallelization of high-level primitives (e.g., tiling for thread- and task-level parallelism).
//!
//! Through the use of procedural macros, this crate exposes to the user a Rust based eDSL 
//! which will be lowered to various IR dialects and then compiled into a shared library at compile time of the user application
//! At runtime the shared library is dynamically linked to, exposing the compiled COMET functions, allowing the user to execute them as the would any other function.
//! 
//! # External Dependencies
//! - [COMET](https://github.com/pnnl/COMET)
//!
//! Please follow the build instructions in the COMET repository to install the COMET compiler.
//! This crate will not work without a successful COMET installation.
//! 
//! # Cargo.toml
//! ```toml
//! [dependencies]
//! comet-rs = "0.1.0"
//!```
//! # Required Environment Variables
//! ```text
//! COMET_BIN_DIR = /path/to/COMET/bin/dir
//! COMET_LIB_DIR = /path/to/COMET/lib/dir
//! MLIR_BIN_DIR = /path/to/MLIR/bin/dir
//! MLIR_LIB_DIR = /path/to/MLIR/lib/dir
//! ```
//! *Note that as part of the COMET build process we will also build a specific version of MLIR (managed as a git submodule),
//! COMET will only work with this specific commit, so please do not point to a different MLIR version you mave have build outside the COMET build process.*
//!
//! # Example
//!
//! ```
//! use comet_rs::*;
//!
//! comet_fn! { dense_dense_matrix_multiply, {
//!    let a = Index::with_value(2);
//!    let b = Index::with_value(2);
//!    let c = Index::with_value(2);
//!
//!    let A = Tensor::<f64>::dense([a, b]).fill(2.2);
//!    let B = Tensor::<f64>::dense([b, c]).fill(3.4);
//!    let C = Tensor::<f64>::dense([a, c]).fill(0.0);
//!    C = A * B;
//!    C.print();
//! }}
//!
//! fn main() {
//!     dense_dense_matrix_multiply();
//! }
//! ```
//! # Operations
//! 
//! We have implemented the following tensor operations (most of which are not valid rust syntax, but are valid COMET eDSL syntax)
//! please refer to the [COMET documentation](https://pnnl-comet.readthedocs.io/en/latest/operations.html) for more in-depth descriptions of each operation.
//! - Multiplication: `A * B`
//! - Elementwise Multiplication: `A .* B`
//! - Semiring Operations: `@(op1, op2)`
//!   - Min: ` A @(min) B`
//!   - Plus: `A @(+) B`
//!   - Mul: `A @(*) B`
//!   - Any-Pair: `A @(any,pair) B`
//!   - Plus-Mul: `A @(+,*)B `
//!   - Plus-Pair: `A @(+,pair) B`
//!   - Plus-First: `A @(+,first) B`
//!   - Plus-Second: `A @(+,Second) B`
//!   - Min-Plus: `A @(min,+) B` 
//!   - Min-First: `A @(min,first) B`
//!   - Min-Second: `A @(min,second) B`
//! - Transpose: `B = A.transpose()`
//! 
//! # Optimizations
//! We also support the ability to specify various optimizations to be performed by the COMET compiler.
//! please refer to the [COMET documentation](https://pnnl-comet.readthedocs.io/en/latest/optimizations.html) for more in-depth descriptions of each optimization.
//! - [Permutation TTGT](https://pnnl-comet.readthedocs.io/en/latest/passes/PermTTGT.html): `BestPermTtgt`
//! - [Tensor Algebra to Index Tree](https://pnnl-comet.readthedocs.io/en/latest/passes/TAtoIT.html): `TaToIt`
//! - [Tensor Contraction to TTGT](https://pnnl-comet.readthedocs.io/en/latest/passes/TC.html): `TcToTtgt`
//! - [Loops](https://pnnl-comet.readthedocs.io/en/latest/passes/loops.html): `ToLoops`
//! - [Matmult Kernel](https://pnnl-comet.readthedocs.io/en/latest/passes/mkernel.html): `MatMulKernel`
//! - [Matmult Tiling](https://pnnl-comet.readthedocs.io/en/latest/passes/tiling.html): `MatMulTiling`
//! - [Dense Transpose](https://pnnl-comet.readthedocs.io/en/latest/passes/transpose.html): `DenseTranspose`
//! - [Workspace](https://pnnl-comet.readthedocs.io/en/latest/passes/workspace.html): `CompWorkspace`
//!
//! The above optimizations can be passed to the compiler as part of a custom syntax proivded as an argument to the `comet_fn` macro. 
//!
//! #### Example
//! ```
//! comet_fn! {function_name, {
//!         eDSL code
//! },
//! CometOptions::[TcToTtgt, BestPermTtgt, ToLoop]
//! }
//! ``` 
//!
//! # COMET Output
//! During evaluation of the `comet_fn` macro, the COMET compiler will generate a shared library containing the compiled COMET function.
//! The shared library will be located in the same directory as the user application in a directory labelled `comet_libs`.
//! This crate handles linking this shared library into your application automatically, but it depends on the library remainng in the `comet_libs` directory.


use comet_rs_impl;

/// COMET uses Einstein mathematical notation and The `comet_fn!` macro provides users with an interface to express tensor algebra semantics using a Rust-like eDSL.
///
/// # Syntax
/// * `function_name` - is the (user defined) name of the COMET function and can be called directly from the rust program as `function_name()`.
/// * `eDSL code` - is the eDSL code to be compiled by the COMET compiler.
/// * `CometOptions` - are optional optimizations to pass to the COMET compiler.
/// ```
/// comet_rs::comet_fn! {function_name, {
/// eDSL code
/// }[,]
/// [CometOptions::[options],]
/// }
/// ``` 
/// # Example
/// ```
/// comet_rs::comet_fn! { print_dense, {
///    let i = Index::with_value(4);
///    let j = Index::with_value(4);
///
///    let A = Tensor::<f64>::dense([i, j]).fill(2.3);
///
///    A.print();
///}}
///
///comet_rs::comet_fn! { sum_coo, {
///    let i = Index::new();
///    let j = Index::new();
///
///    let A = Tensor::<f64>::coo([i, j]).load("../../../integration_test/data/test.mtx");
///    let a = A.sum();
///    a.print();
///}}
///
///comet_rs::comet_fn! { ccsd_t1_4_ttgt_bestperm, {
///    let i, c = Index::with_value(2);
///    let m, a = Index::with_value(4);
///
///    let v = Tensor::<f64>::dense([c, i, m, a]).fill(2.3);
///    let t2 = Tensor::<f64>::dense([m, c]).fill(3.4);
///    let i0 = Tensor::<f64>::dense([i, a]).fill(0.0);
///    i0 = v * t2;
///    i0.print();
///},
///CometOption::[BestPermTtgt, TcToTtgt]
///}
///
///fn main() {
///    print_dense();
///    sum_coo();
///    ccsd_t1_4_ttgt_bestperm();
///}
/// ```

pub use comet_rs_impl::comet_fn; //,main,function};

use std::sync::Arc;

use std::collections::{HashMap, HashSet};

#[doc(hidden)]
pub use inventory;

use ctor::ctor;

#[doc(hidden)]
#[ctor]
pub static COMET_FUNCS: HashMap<
    &'static str,
    (
        libloading::os::unix::Symbol<unsafe extern "C" fn()>,
        Arc<libloading::Library>,
    ),
> = {
    let mut shared_libs = HashSet::new();
    for lib in crate::inventory::iter::<CometLib> {
        shared_libs.insert(lib.name);
    }
    let mut libs = vec![];
    for lib in shared_libs {
        libs.push(Arc::new(libloading::Library::new(lib).unwrap()));
    }
    let mut func_names = HashSet::new();
    for func in crate::inventory::iter::<CometFunc> {
        func_names.insert(func.name);
    }
    let mut map = HashMap::new();
    for func in func_names {
        let func_ptr = libs
            .iter()
            .filter_map(
                |lib| match lib.get::<unsafe extern "C" fn()>(func.as_bytes()) {
                    Ok(func_ptr) => Some((func_ptr.into_raw(), lib.clone())),
                    Err(_) => None,
                },
            )
            .last()
            .expect("function not found");
        map.insert(func.clone(), func_ptr);
    }
    map
};

#[doc(hidden)]
#[derive(Debug)]
pub struct CometFunc {
    pub name: &'static str,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CometLib {
    pub name: &'static str,
}

crate::inventory::collect!(CometFunc);
crate::inventory::collect!(CometLib);

pub mod index;
pub mod tensor;


