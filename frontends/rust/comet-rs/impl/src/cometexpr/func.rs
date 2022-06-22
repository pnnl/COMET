use crate::{cometexpr::*, CometVars};
use syn::parse::{ParseStream, Result};
use syn::Ident;
use proc_macro_error::abort;
use syn::parenthesized;
use syn::parse::discouraged::Speculative;

#[derive(Debug, Clone)]
pub(crate) struct CometFunc {
    pub operand: Box<CometExpr>,
    pub op: CometFuncOp,
}

#[derive(Debug, Clone)]
pub(crate) enum CometFuncOp {
    Print,
    Fill(TensorFill),
    GetTime,
    PrintElapsedTime(CometScalar,CometScalar)
}

impl CometDsl for CometFunc {
    fn emit_mlir(&mut self) -> Result<String> {
        // println!("emit_mlir CometTranspose");
        let res;
        if let Some(_) = self.operand.tensor_mut() {
            let temp = self.tensor_emit_mlir()?;
            res = format!("{}", temp);
        } else if let Some(scalar) = self.operand.scalar() {
            res = format!("{}", self.scalar_emit_mlir(scalar)?);
        } 
        else {
            res = format!("{}", self.no_operand_emit_mlir()?);
        }
        // else{
        //     abort!(self.operand.span(), "Unsupported operation");
        // }
        Ok(res)
    }
    fn my_parse(input: ParseStream, _vars: &mut CometVars, _object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometFunc {input}");
        let fork = input.fork();
        if let Ok(fill) = fork.parse::<TensorFill>() {
            input.advance_to(&fork);
            return Ok(CometFunc {
                operand: Box::new(CometExpr::None),
                op: CometFuncOp::Fill(fill),
            });
        }
        match fork.parse::<Ident>() {
            Ok(ident) => match ident.to_string().as_str() {
                "print" => {
                    let _content;
                    parenthesized!(_content in fork);
                    fork.parse::<Token![;]>()?;
                    input.advance_to(&fork);
                    Ok(CometFunc {
                        operand: Box::new(CometExpr::None),
                        op: CometFuncOp::Print,
                    })
                },
                "getTime" => {
                    let _content;
                    parenthesized!(_content in fork);
                    input.advance_to(&fork);
                    Ok(CometFunc {
                        operand: Box::new(CometExpr::None),
                        op: CometFuncOp::GetTime,
                    })
                },
                "printElapsedTime" => {
                    let content;
                    parenthesized!(content in fork);
                    let mut scalars = vec![];
                    while !content.is_empty() {
                        match CometScalar::my_parse(&content, _vars, _object_id) {
                            Ok(scalar) => {
                                scalars.push(scalar);
                            }
                            Err(_) => {
                                abort!(content.span(), "expected scalar");
                            }
                        }
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                        }
                    }
                    if scalars.len() != 2 {
                        abort!(input.span(), "expected 2 scalars");
                    }
                    fork.parse::<Token![;]>()?;
                    input.advance_to(&fork);
                    Ok(
                        CometFunc {
                            operand: Box::new(CometExpr::None),
                            op: CometFuncOp::PrintElapsedTime(scalars[0].clone(), scalars[1].clone()),
                        }
                    )
                }
                _ => Err(Error::new(input.span(), "Unsupported operation")),
            },
            _ => {
                abort!(input.span(), "Expected an identifier after '.'")
            }
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        self.operand.tensor_mut()
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        self.operand.tensor()
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        self.operand.scalar_mut()
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        self.operand.scalar()
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.operand.set_mlir_id(id);
    }
    fn span(&self) -> Span {
        self.operand.span()
    }
}

impl CometFunc {
    fn tensor_emit_mlir(&mut self) -> Result<String> {
        let tensor = self.operand.tensor_mut().unwrap();
        let (_, dims) = tensor.index_ids_and_dims();
        match &self.op {
            CometFuncOp::Fill(fill) => {
                if tensor.mutable {
                    tensor.fill = fill.clone();
                    let res = format!(
                        "{} : ({}) -> ()\n\n",
                        fill.emit_mlir(tensor.mlir_id, &tensor.ty),
                        dims
                    );
                    Ok(res)
                } else {
                    abort!(fill.span(), "Tensor is not mutable");
                }
            }
            CometFuncOp::Print => {
                let res = format!("\"ta.print\"(%{}) : ({}) -> ()\n\n", tensor.mlir_id, dims);
                Ok(res)
            }
            _ => {
                abort!(self.span(), "Unsupported tensor operation");
            }
        }
    }
    fn scalar_emit_mlir(&self, scalar: &ScalarStruct) -> Result<String> {
        match &self.op {
            CometFuncOp::Fill(fill) => {
                abort!(fill.span(), "fill not valid method for scalar");
            }
            CometFuncOp::Print => {
                let res = format!(
                    "\"ta.print\"(%{}) : ({}) -> ()\n\n",
                    scalar.mlir_id, scalar.ty
                );
                Ok(res)
            }
            CometFuncOp::GetTime => {
                let res = format!("%{} = \"ta.getTime\"() : () -> f64\n\n",scalar.mlir_id);
                Ok(res)
            }
            _ => {
                abort!(self.span(), "Unsupported scalar operation");
            }
        }
    }
    fn no_operand_emit_mlir(&self) -> Result<String> {
        match &self.op{
            CometFuncOp::PrintElapsedTime(start, end) => {
                let res = format!(
                    "\"ta.print_elapsed_time\"(%{}, %{}) : (f64,f64) -> ()\n\n",
                    start.scalar.mlir_id, end.scalar.mlir_id
                );
                Ok(res)
            }
            _ => {
                abort!(self.span(), "Unsupported operation");
            }
        }
    }
}
