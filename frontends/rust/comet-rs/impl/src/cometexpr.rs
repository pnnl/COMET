use crate::{index::*, scalar::*, tensor::*, CometVars};
use proc_macro2::{Span, TokenStream, TokenTree};
use proc_macro_error::{abort, abort_if_dirty};
use quote::{ToTokens,format_ident};
use std::str::FromStr;
use syn::parse::discouraged::Speculative;
use syn::parse::Parser;
use syn::parse::{ ParseStream, Result};
use syn::Token;
use syn::{Error, ExprBinary, /*LitFloat, LitInt*/};

pub(crate) mod assign;
use assign::*;
pub(crate) mod binary;
use binary::*;
pub(crate) mod func;
use func::*;
pub(crate) mod scalar;
use scalar::*;
pub(crate) mod tensor;
use tensor::*;
pub(crate) mod unary;
use unary::*;

// #[derive(Debug, Clone)]
// pub(crate) enum CometLiteral {
//     Int(LitInt),
//     Float(LitFloat),
// }

#[derive(Debug, Clone)]
pub(crate) enum CometExpr {
    Tensor(CometTensor),
    Scalar(CometScalar),
    Assign(CometAssign),
    Binary(CometBinary),
    Unary(CometUnary),
    Func(CometFunc),
    // Lit(LitFloat),
    None,
}

pub(crate) trait CometDsl {
    fn emit_mlir(&mut self) -> Result<String>;
    fn my_parse(input: ParseStream, vars: &mut CometVars, object_id: &mut usize) -> Result<Self>
    where
        Self: Sized;
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct>;
    fn tensor(&self) -> Option<&TensorStruct>;
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct>;
    fn scalar(&self) -> Option<&ScalarStruct>;
    fn set_mlir_id(&mut self, id: usize);
    fn span(&self) -> Span;
}

// impl CometLiteral {
//     pub(crate) fn span(&self) -> Span {
//         match self {
//             CometLiteral::Int(i) => i.span(),
//             CometLiteral::Float(f) => f.span(),
//         }
//     }
// }

impl CometDsl for CometExpr {
    fn emit_mlir(&mut self) -> Result<String> {
        // println!("emit_mlir CometExpr ");
        match self {
            // the below should never acually called
            CometExpr::Tensor(_) => {
                //the actuall tensor is emited where declared
                let res = String::new();
                Ok(res)
            }
            CometExpr::Scalar(_) => {
                let res = String::new();
                Ok(res)
            }
            CometExpr::Assign(assign) => {
                let res = assign.emit_mlir();
                res
            }
            CometExpr::Binary(bin) => {
                let res = bin.emit_mlir();
                res
            }
            CometExpr::Unary(unary) => {
                let res = unary.emit_mlir();
                res
            }
            CometExpr::Func(func) => {
                let res = func.emit_mlir();
                res
            }
            // CometExpr::Lit(_) => {
            //     let res = String::new();
            //     Ok(res)
            // }
            CometExpr::None => {
                abort!(Span::call_site(), "CometExpr::None should not be emitted");
            }
        }
    }

    fn my_parse(input: ParseStream, vars: &mut CometVars, object_id: &mut usize) -> Result<Self> {
        let fork = input.fork();
        // println!("0. parse CometExpr  {fork}");

        if let Ok(expr) = CometExpr::parse_custom_ops(&fork, vars, object_id) {
            input.advance_to(&fork);
            return Ok(expr);
        }
        // println!("1. parse CometExpr  {fork}");

        let fork2 = fork.fork();
        if let Ok(tensor) = CometTensor::my_parse(&fork2, vars, object_id) {
            if fork2.is_empty() {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(CometExpr::Tensor(tensor));
            }
            if let Ok(res) = CometExpr::parse_tensor_method(tensor.clone(), &fork2, vars, object_id)
            {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(res);
            }
            if let Ok(res) = CometExpr::parse_tensor_assign(tensor.clone(), &fork2, vars, object_id)
            {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(res);
            }
        }
        // println!("2. parse CometExpr  {fork}");
        let fork2 = fork.fork();
        if let Ok(scalar) = CometScalar::my_parse(&fork2, vars, object_id) {
            if fork2.is_empty() {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(CometExpr::Scalar(scalar));
            }
            if let Ok(res) = CometExpr::parse_scalar_method(scalar.clone(), &fork2, vars, object_id)
            {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(res);
            }
            if let Ok(res) = CometExpr::parse_scalar_assign(scalar.clone(), &fork2, vars, object_id)
            {
                fork.advance_to(&fork2);
                input.advance_to(&fork2);
                return Ok(res);
            }
        }
        // println!("3. parse CometExpr  {fork}");

        if let Ok(func_expr) = CometFunc::my_parse(&fork, vars, object_id) {
            input.advance_to(&fork);
            return Ok(CometExpr::Func(func_expr));
        }

        // println!("4. parse CometExpr  {fork}");

        // let fork = input.fork();
        if let Ok(bin_expr) = fork.parse::<ExprBinary>() {
            //let syn do the heavy work of figuring out the binary expressions, although for now I think its only a single layer...
            match CometExpr::parse_comet_binary(bin_expr, &fork, vars, object_id) {                
                Ok(res) => {
                    input.advance_to(&fork);
                    return Ok(res);
                }
                Err(e) => {
                    println!("parse_comet_binary error {:?}", e);
                    return Err(e);
                }
            }
        } else {
            // println!("5. parse CometExpr  {fork}");
            abort_if_dirty();
            abort!(input.span(), "Unsupported operation or undeclared variable");
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        match self {
            CometExpr::Tensor(tensor) => tensor.tensor_mut(),
            CometExpr::Scalar(scalar) => scalar.tensor_mut(),
            CometExpr::Assign(assign) => assign.tensor_mut(),
            CometExpr::Binary(bin) => bin.tensor_mut(),
            CometExpr::Unary(unary) => unary.tensor_mut(),
            CometExpr::Func(func) => func.tensor_mut(),
            // CometExpr::Lit(_) => None,
            CometExpr::None => None,
        }
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        match self {
            CometExpr::Tensor(tensor) => tensor.tensor(),
            CometExpr::Scalar(scalar) => scalar.tensor(),
            CometExpr::Assign(assign) => assign.tensor(),
            CometExpr::Binary(bin) => bin.tensor(),
            CometExpr::Unary(unary) => unary.tensor(),
            CometExpr::Func(func) => func.tensor(),
            // CometExpr::Lit(_) => None,
            CometExpr::None => None,
        }
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        match self {
            CometExpr::Tensor(tensor) => tensor.scalar_mut(),
            CometExpr::Scalar(scalar) => scalar.scalar_mut(),
            CometExpr::Assign(assign) => assign.scalar_mut(),
            CometExpr::Binary(bin) => bin.scalar_mut(),
            CometExpr::Unary(unary) => unary.scalar_mut(),
            CometExpr::Func(func) => func.scalar_mut(),
            // CometExpr::Lit(_) => panic!("CometExpr::Lit should not be used as a scalar"),
            CometExpr::None => None,
        }
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        match self {
            CometExpr::Tensor(tensor) => tensor.scalar(),
            CometExpr::Scalar(scalar) => scalar.scalar(),
            CometExpr::Assign(assign) => assign.scalar(),
            CometExpr::Binary(bin) => bin.scalar(),
            CometExpr::Unary(unary) => unary.scalar(),
            CometExpr::Func(func) => func.scalar(),
            // CometExpr::Lit(_) => panic!("CometExpr::Lit should not be used as a scalar"),
            CometExpr::None => None,
        }
    }
    fn set_mlir_id(&mut self, id: usize) {
        match self {
            CometExpr::Tensor(tensor) => tensor.set_mlir_id(id),
            CometExpr::Scalar(scalar) => scalar.set_mlir_id(id),
            CometExpr::Assign(assign) => assign.set_mlir_id(id),
            CometExpr::Binary(bin) => bin.set_mlir_id(id),
            CometExpr::Unary(unary) => unary.set_mlir_id(id),
            CometExpr::Func(func) => func.set_mlir_id(id),
            // CometExpr::Lit(_) => panic!("CometExpr::Lit does not have mlir_id"),
            CometExpr::None => panic!("CometExpr::None does not have mlir_id"),
        }
    }
    fn span(&self) -> Span {
        match self {
            CometExpr::Tensor(tensor) => tensor.span(),
            CometExpr::Scalar(scalar) => scalar.span(),
            CometExpr::Assign(assign) => assign.span(),
            CometExpr::Binary(bin) => bin.span(),
            CometExpr::Unary(unary) => unary.span(),
            CometExpr::Func(func) => func.span(),
            // CometExpr::Lit(lit) => lit.span(),
            CometExpr::None => panic!("CometExpr::None does not have span"),
        }
    }
}

impl CometExpr {
    fn parse_custom_ops(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse_custom_ops {input}");
        let fork = input.fork();
        match fork.step(|cursor| {
            //check for elementwise mul
            let mut tts = Vec::new();
            let mut rest = *cursor;
            let mut real_rest = *cursor;
            let mut semi = false;
            while let Some((tt, next)) = rest.token_tree() {
                if !semi {
                    match &tt {
                        TokenTree::Punct(punc) => {
                            if punc.as_char() == ';' {
                                real_rest = next.clone();
                                semi = true;
                            }
                        }
                        _ => {}
                    }
                }
                tts.push(tt);
                rest = next;
            }
            let token_stream: TokenStream = tts.into_iter().collect();
            let as_str = token_stream.to_string();
            let as_str = as_str.split(";").collect::<Vec<&str>>();
            if as_str.len() > 0 {
                let as_str = as_str[0].to_string() + ";";
                if let Some(_) = as_str.find(".*") {
                    //because .* is not a standard operator we convert to an operator unused for tensor algebras
                    vars.custom_ops.insert("|".to_string(),".*".to_string());
                    let as_str = as_str.replace(".*", "|"); //this is to let syn approriately parse the expression using accecpted operators
                    let tok_stream = TokenStream::from_str(&as_str).unwrap();
                    let parser = |the_input: ParseStream| {
                        let res: Result<CometExpr> =
                            CometExpr::my_parse(the_input, vars, object_id);
                        res
                    };
                    match parser.parse2(tok_stream) {
                        Ok(res) => {
                            return Ok((res, real_rest));
                        }
                        Err(_) => {
                            return Err(cursor.error("error parsing CometExpr"));
                        }
                    }
                }
                if let Some(_) = as_str.find("@") {
                    let re = regex::Regex::new(r#"@\s*\((.*)\)"#).unwrap();
                    let op = re.captures(&as_str).unwrap().get(1).unwrap().as_str();
                    let split_str = re.split(&as_str).collect::<Vec<&str>>();
                    let new_str = split_str[0..(split_str.len() - 1)].iter().fold(String::new(), |acc, x| acc + x + " & ") + split_str[split_str.len() - 1];
                    //because .* is not a standard operator we convert to an operator unused for tensor algebras
                    vars.custom_ops.insert("&".to_string(),op.to_string());
                    // let as_str = as_str.replace(".*", "|"); //this is to let syn approriately parse the expression using accecpted operators
                    let tok_stream = TokenStream::from_str(&new_str).unwrap();
                    let parser = |the_input: ParseStream| {
                        let res: Result<CometExpr> =
                            CometExpr::my_parse(the_input, vars, object_id);
                        res
                    };
                    match parser.parse2(tok_stream) {
                        Ok(res) => {
                            return Ok((res, real_rest));
                        }
                        Err(_) => {
                            return Err(cursor.error("error parsing CometExpr"));
                        }
                    }
                }
            }
            return Err(cursor.error("no .* operator found"));
        }) {            
            Ok(res) => {
                input.advance_to(&fork);
                return Ok(res);
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    fn parse_tensor_method(
        tensor: CometTensor,
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse_tensor_method {input}");
        let fork = input.fork();
        if fork.peek(Token![.]) {
            fork.parse::<Token![.]>()?;
            if let Ok(mut func) = CometFunc::my_parse(&fork, vars, object_id) {
                if fork.peek(Token![;]) {
                    fork.parse::<Token![;]>()?;
                }
                input.advance_to(&fork);
                func.operand = Box::new(CometExpr::Tensor(tensor));
                return Ok(CometExpr::Func(func));
            } else if let Ok(mut unary) = CometUnary::my_parse(&fork, vars, object_id) {
                input.advance_to(&fork);
                unary.operand = Box::new(CometExpr::Tensor(tensor));
                return Ok(CometExpr::Unary(unary));
            } else {
                return Err(Error::new(fork.span(), "Unexpected method call"));
            }
        } else {
            return Err(Error::new(input.span(), "not a method call"));
        }
    }

    fn parse_tensor_assign(
        left: CometTensor,
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse_tensor_assign {input}");
        let fork = input.fork();
        if fork.peek(Token![=]) {
            fork.parse::<Token![=]>()?;
            if let Ok(mut assign) = CometAssign::my_parse(&fork, vars, object_id) {
                if let Err(_e) = fork.parse::<Token![;]>() {
                    abort!(fork.span(), "Expected ; after expression");
                }
                input.advance_to(&fork);
                
                match *assign.right {
                    CometExpr::Binary(ref mut bin) => {
                        if let CometExpr::None = *bin.root {
                            // let mut temp = left.clone();
                            // temp.set_mlir_id(*object_id);
                            // *object_id += 1;
                            // bin.root = Box::new(CometExpr::Tensor(temp));
                            bin.assign_tensor_root(left.clone(),vars,object_id);
                        }
                        
                    }
                    CometExpr::Unary(ref mut unary) => {
                        if let CometExpr::None = *unary.imm_expr{
                            let mut temp = left.clone();
                            temp.set_mlir_id(*object_id);
                            *object_id += 1;
                            unary.imm_expr = Box::new(CometExpr::Tensor(temp));
                        }
                    }
                    _ => {
                        abort!(fork.span(), "Expected binary or unary expression");
                    }
                }
                assign.left = Box::new(CometExpr::Tensor(left));
                Ok(CometExpr::Assign(assign))
            } else {
                return Err(Error::new(fork.span(), "Expected expression after ="));
            }
        } else {
            return Err(Error::new(input.span(), "not an assignment"));
        }
    }

    fn parse_comet_binary(
        bin_expr: ExprBinary,
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse Binary {input}");
        let fork = input.fork();
        let lparser = |linput: ParseStream| CometExpr::my_parse(linput, vars, object_id);
        let left = Box::new(lparser.parse2(bin_expr.left.to_token_stream())?);
        let opparser = |oinput: ParseStream| CometBinOp::my_parse(oinput, vars);
        let op = opparser.parse2(bin_expr.op.to_token_stream())?;
        let rparser = |rinput: ParseStream| CometExpr::my_parse(rinput, vars, object_id);
        let right = Box::new(rparser.parse2(bin_expr.right.to_token_stream())?);
        let root;
        // if left.tensor().is_some() && right.tensor().is_some() {
        //     let mut tensor = left.tensor().unwrap().clone();
        //     tensor.name = format_ident!("temp_tensor_{}",*object_id);
        //     tensor.fill = TensorFill::None;
        //     tensor.mlir_id = *object_id;
        //     *object_id += 1;
        //     vars.tensors
        //         .entry(tensor.name.clone())
        //         .and_modify(|e| *e = tensor.clone())
        //         .or_insert(tensor.clone());
        //     root = Box::new(CometExpr::Tensor(CometTensor{name: tensor.name.clone(),tensor: tensor}));
        // }
        // else {
            root = Box::new(CometExpr::None);
        // }
        let res = CometExpr::Binary(CometBinary {
            root: root,
            left: left,
            op: op, //syn::parse2::<CometBinOp>(bin_expr.op.to_token_stream())?,
            right: right,
        });
        
        // println!("parse Binary {:?}",res);
        input.advance_to(&fork);
        return Ok(res);
    }

    fn parse_scalar_method(
        scalar: CometScalar,
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse_scalar_method {input}");
        let fork = input.fork();
        if fork.peek(Token![.]) {
            fork.parse::<Token![.]>()?;
            if let Ok(mut func) = CometFunc::my_parse(&fork, vars, object_id) {
                if fork.peek(Token![;]) {
                    fork.parse::<Token![;]>()?;
                }
                input.advance_to(&fork);
                func.operand = Box::new(CometExpr::Scalar(scalar));
                return Ok(CometExpr::Func(func));
            } else if let Ok(mut unary) = CometUnary::my_parse(&fork, vars, object_id) {
                input.advance_to(&fork);
                unary.operand = Box::new(CometExpr::Scalar(scalar));
                return Ok(CometExpr::Unary(unary));
            } else {
                return Err(Error::new(fork.span(), "Unexpected method call"));
            }
        } else {
            return Err(Error::new(input.span(), "not a method call"));
        }
    }

    fn parse_scalar_assign(
        left: CometScalar,
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        // println!("parse_scalar_assign {input}");
        let fork = input.fork();
        if fork.peek(Token![=]) {
            fork.parse::<Token![=]>()?;
            if let Ok(mut assign) = CometAssign::my_parse(&fork, vars, object_id) {
                if let Err(_e) = fork.parse::<Token![;]>() {
                    abort!(fork.span(), "Expected ; after expression");
                }
                input.advance_to(&fork);
                let temp = left.clone();
                match *assign.right {
                    CometExpr::Binary(ref mut bin) => {
                        bin.root = Box::new(CometExpr::Scalar(temp));
                    }
                    CometExpr::Unary(ref mut unary) => {
                        unary.imm_expr = Box::new(CometExpr::Scalar(temp));
                    }
                    CometExpr::Func(ref mut func) => {
                        func.operand = Box::new(CometExpr::Scalar(temp));
                    }
                    // CometExpr::Lit(_) => {
                    //     abort!(fork.span(), "Need to handle scalar literal assignment");
                    // }
                    _ => {
                        abort!(fork.span(), "Expected binary or unary expression");
                    }
                }
                assign.left = Box::new(CometExpr::Scalar(left));
                Ok(CometExpr::Assign(assign))
            } else {
                return Err(Error::new(fork.span(), "Expected expression after ="));
            }
        } else {
            return Err(Error::new(input.span(), "not an assignment"));
        }
    }
}
