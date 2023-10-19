extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro_error::{proc_macro_error,abort};
use quote::quote;
use std::collections::HashMap;
use syn::{braced, Attribute, Ident, Stmt, Token};

mod index;
use index::IndexStruct;

mod tensor;
use tensor::*;

mod scalar;
use scalar::ScalarStruct;

mod cometexpr;
use cometexpr::*;

use std::env;
use std::path::PathBuf;
use std::process::Command;

enum CometResult{
    Success(String),
    Failure(String),    
}

// fn create_lib(func_name: &str, mlir_str: &str, comet_opts: Vec<CometOption>, mlir_opts: Vec<MlirOption>) -> CometResult{
fn create_lib(func_name: &str, mlir_str: &str, comet_opts: Vec<CometOption>) -> CometResult{
    // println!("{}",env::var("CARGO_CRATE_NAME").unwrap());
    // for (key,value) in env::vars() {
    //     println!("{}={}",key,value);
    // }

    let comet_base = if let Ok(base) = env::var("COMET_DIR") {
        PathBuf::from(base)
    } else {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../..")
    };

    let comet_bin = if let Ok(dir) = env::var("COMET_BIN_DIR"){
        PathBuf::from(dir).join("comet-opt")
    }
    else{
        comet_base.clone().join("build/bin/comet-opt")
    };
    if !comet_bin.exists(){
        panic!("Cannot find comet-opt at {}\n Please set the COMET_BIN_DIR envrionement variable to the directory containing the comet-opt binary", comet_bin.display())
    }

    let comet_lib = if let Ok(dir) = env::var("COMET_LIB_DIR"){
        PathBuf::from(dir)
    }
    else{
        comet_base.clone().join("build/lib")
    };
    if !comet_lib.exists(){
        panic!("Cannot find comet lib directory at {}\n Please set the COMET_LIB_DIR envrionement variable to the directory containing the comet shared libraries", comet_lib.display())
    }

    let mlir_bin = if let Ok(dir) = env::var("MLIR_BIN_DIR"){
        PathBuf::from(dir)
    }
    else{
        comet_base.clone().join("llvm/build/bin")
    };
    if !mlir_bin.exists(){
        panic!("Cannot find mlir-opt at {}\n Please set the MLIR_OPT_DIR envrionement variable to the directory containing the mlir-opt binary", mlir_bin.display())
    }

    let mlir_lib = if let Ok(dir) = env::var("MLIR_LIB_DIR"){
        PathBuf::from(dir)
    }
    else{
        comet_base.clone().join("llvm/build/lib")
    };
    if !mlir_lib.exists(){
        panic!("Cannot find mlir lib directory at {}\n Please set the MLIR_LIB_DIR envrionement variable to the directory containing the mlir shared libraries", mlir_lib.display())
    }


    let base_libpath = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("comet_libs")
        .join(env::var("CARGO_CRATE_NAME").unwrap());
    std::fs::create_dir_all(base_libpath.clone()).expect("Could not create directory");
    let base_filepath = base_libpath.clone().join(func_name);

    let comet_mlir_file = base_filepath.clone().with_extension("comet.mlir");
    std::fs::write(&comet_mlir_file, mlir_str).expect("Unable to write file");
    // println!("{}", comet_mlir_file.to_str().unwrap());

    let mut cmd_str = String::new();
    let mut cmd = Command::new(comet_bin);
    for opt in comet_opts {
        cmd_str += opt.as_str();
        cmd.arg(opt.as_str());
    }
    let output = cmd
        .arg(comet_mlir_file.to_str().unwrap())
        .output()
        .expect("Could not run comet-opt");
    


    if !output.status.success() {
        if cfg!(feature = "comet_errors_as_warnings") {
            println!("ERROR: failed to convert from rs to mlir {:?} option str {}", String::from_utf8_lossy(&output.stderr),cmd_str);
            return CometResult::Failure(format!("ERROR: failed to convert from rs to mlir {:?} option str {}", String::from_utf8_lossy(&output.stderr),cmd_str));
        }
        else  {
            panic!("ERROR: failed to convert from rs to mlir {:?} option str {}", String::from_utf8_lossy(&output.stderr),cmd_str);
        }
    }
    let llvm_file = base_filepath.clone().with_extension("llvm");
    std::fs::write(&llvm_file, &output.stderr).expect("Unable to write file");

    let bc_file = base_filepath.clone().with_extension("bc");
    let _status = Command::new(mlir_bin.clone().join("mlir-translate"))
        .arg("--mlir-to-llvmir")
        .arg(llvm_file.to_str().unwrap())
        .arg("-o")
        .arg(bc_file.to_str().unwrap())
        .status()
        .expect("Could not run mlir-translate");

    let obj_file = base_filepath.clone().with_extension("comet.o");
    let _status = Command::new(mlir_bin.clone().join("llc"))
        .arg("-O3")
        .arg(bc_file.to_str().unwrap())
        .arg("-o")
        .arg(obj_file.to_str().unwrap())
        .arg("-filetype=obj")
        .status()
        .expect("Could not run llc");

    let so_file = base_filepath.clone().with_extension("so");
    // let llvm_lib = comet_base.clone().join("llvm/build/lib").into_os_string().into_string().unwrap();
    // let comet_lib = comet_base.clone().join("build/lib").into_os_string().into_string().unwrap();
    let _status = Command::new("gcc")
        .arg(format!{"-Wl,-rpath,{}", mlir_lib.display()})
        .arg(format!{"-L{}", mlir_lib.display()})
        .arg("-lmlir_runner_utils")
        .arg(format!{"-Wl,-rpath,{}",comet_lib.display()})
        .arg(format!{"-L{}",comet_lib.display()})
        .arg("-lcomet_runner_utils")
        .arg("-shared")
        .arg("-o")
        .arg(so_file.to_str().unwrap())
        .arg(obj_file.to_str().unwrap())
        .status()
        .expect("Could not run gcc");


    CometResult::Success(so_file.to_str().unwrap().to_string())
}

use syn::parse_macro_input;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream};

#[derive(Debug)]
enum Comet {
    Index(Vec<IndexStruct>),
    Tensor(TensorStruct),
    Scalar(ScalarStruct),
    Expr(CometExpr),
}

#[derive(Debug)]
pub(crate) struct CometVars {
    indices: HashMap<Ident, IndexStruct>,
    tensors: HashMap<Ident, TensorStruct>,
    scalars: HashMap<Ident, ScalarStruct>,
    custom_ops: HashMap<String, String>,
}

impl Comet {
    fn my_parse(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> syn::Result<Self> {
        let mut errs = vec![];
        match IndexStruct::my_parse(input, vars, object_id) {
            Ok(indices) => {
                return Ok(Comet::Index(indices));
            }
            Err(e) => errs.push(e),
        }
        match TensorStruct::my_parse(input, vars, object_id) {
            Ok(tensor) => {
                return Ok(Comet::Tensor(tensor));
            }
            Err(e) => errs.push(e),
        }

        match ScalarStruct::my_parse(input, vars, object_id) {
            Ok(scalar) => {
                return Ok(Comet::Scalar(scalar));
            }
            Err(e) => errs.push(e),
        }
        match CometExpr::my_parse(input, vars, object_id) {
            Ok(expr) => {
                return Ok(Comet::Expr(expr));
            }
            Err(e) => errs.push(e),
        }
        if errs.len() >= 1 {
            return Err(errs.remove(0));
        } else {
            return Err(syn::Error::new(
                input.span(),
                "Could not parse comet expression",
            )); //this should never happen
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CometOption{
    TcToTtgt,
    ToLoops,
    ToLlvm,
    TaToIt,
    BestPermTtgt,
    MatMulTiling,
    MatMulkernel,
    DenseTranspose,
    CompWorkspace
}

impl CometOption{
    fn as_str(&self) -> &'static str{
        match self{
            CometOption::TcToTtgt =>"--convert-tc-to-ttgt",
            CometOption::ToLoops =>"--convert-to-loops",
            CometOption::TaToIt =>"--convert-ta-to-it",
            CometOption::ToLlvm =>"--convert-to-llvm",
            CometOption::BestPermTtgt =>"-opt-bestperm-ttgt",
            CometOption::MatMulTiling =>"-opt-matmul-tiling",
            CometOption::MatMulkernel =>"-opt-matmul-mkernel",
            CometOption::DenseTranspose =>"-opt-dense-transpose",
            CometOption::CompWorkspace =>"-opt-comp-workspace",
        }
    }
}

impl CometOption {
    fn my_parse(input: ParseStream) -> syn::Result<Vec<Self>> {
        let mut options = vec![];
        let fork = input.fork();
        if let Ok(comet_opt) = fork.parse::<Ident>(){
            if comet_opt.to_string().as_str() == "CometOption"{
                fork.parse::<Token![::]>()?;
                let content;
                syn::bracketed!(content in fork);
                while !content.is_empty() {
                    if let Ok(opt) = content.parse::<Ident>(){                
                        match opt.to_string().as_str() {
                            "TcToTtgt" => {
                                if !options.contains(&CometOption::TcToTtgt){
                                    options.push(CometOption::TcToTtgt);
                                }
                            },
                            "ToLoops" => {
                                if !options.contains(&CometOption::ToLoops){
                                    options.push(CometOption::ToLoops);
                                }
                            },
                            "TaToIt" => {
                                if !options.contains(&CometOption::TaToIt){
                                    options.push(CometOption::TaToIt);
                                }
                            },
                            "ToLlvm" => {
                                if !options.contains(&CometOption::ToLlvm){
                                    options.push(CometOption::ToLlvm);
                                }
                            },
                            "BestPermTtgt" => {
                                if !options.contains(&CometOption::BestPermTtgt){
                                    options.push(CometOption::BestPermTtgt);
                                }
                            },
                            "MatMulTiling" => {
                                if !options.contains(&CometOption::MatMulTiling){
                                    options.push(CometOption::MatMulTiling);
                                }
                            },
                            "MatMulkernel" => {
                                if !options.contains(&CometOption::MatMulkernel){
                                    options.push(CometOption::MatMulkernel);
                                }
                            },
                            "DenseTranspose" => {
                                if !options.contains(&CometOption::DenseTranspose){
                                    options.push(CometOption::DenseTranspose);
                                }
                            },
                            "CompWorkspace" => {
                                if !options.contains(&CometOption::CompWorkspace){
                                    options.push(CometOption::CompWorkspace);
                                }
                            },                        
                            _ => abort!(opt.span(),"Unknown CometOption"),
                        }
                        if content.peek(Token![,]){
                            content.parse::<Token![,]>()?;
                        }
                    }
                    else{
                        abort!(content.span(),"Unknown CometOption");
                    }
                }
                
            }
            else{
                
                return Err(syn::Error::new(
                    comet_opt.span(),
                    "expected CometOption",
                ))
            }
        }
        else{
            return Err(syn::Error::new(
                input.span(),
                "expected CometOption",
            ))
        }
        input.advance_to(&fork);
        Ok(options)
    }
}

pub(crate) struct CometFn {
    pub name: Ident,
    lib_path: CometResult,
    sparse_env: proc_macro2::TokenStream,
}

impl Parse for CometFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;
        let content;
        let _brace_token = braced!(content in input);
        let _inner_attrs = content.call(Attribute::parse_inner)?;
        let mut vars = CometVars {
            indices: HashMap::new(), //index name, mlir_id
            tensors: HashMap::new(), //tensor name, mlir_id
            scalars: HashMap::new(), //scalar name, mlir_id
            custom_ops: HashMap::new(), //custom op name, op str
        };
        let mut object_id = 0;

        let mut mlir_str = format!("module  {{\nfunc.func @{}() {{\n", name);
        while !content.is_empty() {
            let fork = content.fork();
            match Comet::my_parse(&fork, &mut vars, &mut object_id) {
                Ok(comet) => {
                    content.advance_to(&fork);
                    match comet {
                        Comet::Index(indices) => {
                            // println!("index {:?}", index);
                            for index in indices {
                                mlir_str += &index.emit_mlir();
                            }                            
                        }
                        Comet::Tensor(tensor) => {
                            // println!("tensor {:?}", tensor);
                            mlir_str += &tensor.emit_mlir();
                        }
                        Comet::Scalar(mut scalar) => {
                            // println!("scalar {:?}", scalar);
                            mlir_str += &scalar.emit_mlir();
                        }
                        Comet::Expr(mut expr) => {
                            // println!("expr {:?}", expr);
                            mlir_str += &expr.emit_mlir()?;
                        }
                    }
                }
                Err(_) => {
                    let stmt = content.parse::<Stmt>()?;
                    println!("not a comet stmt{:?}", stmt);
                }
            }
        }
        
        mlir_str += "return\n}\n}\n\n";
        // println!("{}", mlir_str);
        let mut opt_comp_workspace = true;
        // let mut sparse_env = String::new();
        let mut sparse_env = quote!{};
        for tensor in vars.tensors.values() {
            // println!("{} {}",tensor.format,tensor.format != TensorFormat::Csr);
            if tensor.format != TensorFormat::Csr {
                opt_comp_workspace = false;
            }

            if let TensorFill::FillFromFile(val,env,_) = &tensor.fill {
                // let stmt = syn::parse_str( &format!("std::env::set_var({},{});\n",env,val.value())).unwrap();
                let p = val.value();
                sparse_env.extend(quote!{std::env::set_var( #env, #p );});
                // sparse_env = env.clone();
                // sparse_env +=;
            }
        }
        if input.peek(Token![,]){
            input.parse::<Token![,]>()?;
        }
        let mut comet_opts = match CometOption::my_parse(&input){
            Ok(opts) => {
                if input.peek(Token![,]){
                    input.parse::<Token![,]>()?;
                }
                opts
            }
            Err(_) => {
                vec![CometOption::TaToIt,CometOption::ToLoops,CometOption::ToLlvm]
            }
        };
       
        if opt_comp_workspace{
            if !comet_opts.contains(&CometOption::CompWorkspace){
                comet_opts.insert(0,CometOption::CompWorkspace);
            }
        }
        // println!("{} {} {:?}",name,opt_comp_workspace,comet_opts);
        let lib_path = create_lib(&name.clone().to_string(), &mlir_str,comet_opts);

        // println!("sparse_env: {}", sparse_env);
        Ok(CometFn {
            name,
            lib_path,
            sparse_env,
        })
    }
}


#[proc_macro_error]
#[proc_macro]
pub fn comet_fn(input: TokenStream) -> TokenStream {

    let func = parse_macro_input!(input as CometFn);

    let name = func.name;
    let sparse_env = func.sparse_env;
    // let sparse_env: Vect<syn::Stmt> = Vec!;
    
    match func.lib_path {
        CometResult::Success(lib_name) => {
            // println!("lib_name: {}", lib_name);
            quote! {
                comet_rs::inventory::submit!{
                    comet_rs::CometFunc{
                        name: stringify!(#name)
                    }

                }
                comet_rs::inventory::submit!{
                    comet_rs::CometLib{
                        name: #lib_name
                    }

                }
                fn #name(){
                    #sparse_env
                    unsafe {comet_rs::COMET_FUNCS.get(stringify!(#name)).unwrap().0();}
                }
            }
            .into()
        }
        CometResult::Failure(msg) =>{
            quote! {
                fn #name(){
                    println!("COMET-RS ERROR: {}",#msg);
                }
            }
            .into()
        }
    }
}
