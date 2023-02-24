use crate::{cometexpr::*, CometVars};
use syn::parse::{ParseStream, Result};
use syn::Ident;
// use syn::{braced,Visibility,Signature,Stmt,Expr,Attribute,Path,token,Item,Block,parse_macro_input};
use proc_macro_error::abort;
use syn::parenthesized;
use syn::parse::discouraged::Speculative;

#[derive(Debug, Clone)]
pub(crate) struct CometUnary {
    pub imm_expr: Box<CometExpr>, //resulting intermediate variable
    pub operand: Box<CometExpr>,
    pub op: CometUnaryOp,
}

#[derive(Debug, Clone)]
pub(crate) enum CometUnaryOp {
    Transpose(IndicesList),
    Sum,
}

impl CometDsl for CometUnary {
    fn emit_mlir(&mut self) -> Result<String> {
        // println!("emit_mlir CometTranspose");
        let res;
        if let Some(tensor) = self.operand.tensor() {
            res = format!("{}", self.tensor_emit_mlir(tensor)?);
        } else if let Some(scalar) = self.operand.scalar() {
            res = format!("{}", self.scalar_emit_mlir(scalar)?);
        } else {
            abort!(self.operand.span(), "Unsupported operation");
        }
        Ok(res)
    }
    fn my_parse(input: ParseStream, vars: &mut CometVars, _object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometUnary {input}");
        let fork = input.fork();
        match fork.parse::<Ident>() {
            Ok(ident) => match ident.to_string().as_str() {
                "transpose" => {
                    let content;
                    parenthesized!(content in fork);
                    let indices_list = IndicesList::my_parse(&content, vars)?;
                    input.advance_to(&fork);
                    return Ok(CometUnary {
                        imm_expr: Box::new(CometExpr::None),
                        operand: Box::new(CometExpr::None),
                        op: CometUnaryOp::Transpose(indices_list),
                    });
                }
                "sum" => {
                    let _content;
                    parenthesized!(_content in fork);
                    input.advance_to(&fork);
                    return Ok(CometUnary {
                        imm_expr: Box::new(CometExpr::None),
                        operand: Box::new(CometExpr::None),
                        op: CometUnaryOp::Sum,
                    });
                }
                _ => {
                    abort!(input.span(), "Unsupported operation");
                }
            },
            _ => {
                abort!(input.span(), "Expected an identifier after '.'")
            }
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        self.imm_expr.tensor_mut()
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        self.imm_expr.tensor()
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        self.imm_expr.scalar_mut()
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        self.imm_expr.scalar()
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.imm_expr.set_mlir_id(id);
    }
    fn span(&self) -> Span {
        self.imm_expr.span()
    }
}

impl CometUnary {
    fn tensor_emit_mlir(&self, tensor: &TensorStruct) -> Result<String> {
        let (_, src_dims) = tensor.index_ids_and_dims();
        match &self.op {
            CometUnaryOp::Transpose(tr_indices) => {
                if let Some(imm_tensor) = self.imm_expr.tensor() {
                    let src_affine = tensor.indices.as_affine();
                    let dst_affine = imm_tensor.indices.as_affine();
                    let (dst_ids, dst_dims) = imm_tensor.index_ids_and_dims();
                    if &imm_tensor.indices != tr_indices {
                        abort!(
                            imm_tensor.indices.span(&imm_tensor.indices[0]),
                            "Transpose indices do not match"
                        );
                    }
                    let res = format!("%{} = \"ta.transpose\"(%{}, {}) {{formats = [\"{}\", \"{}\"], indexing_maps = [ affine_map<({}) -> ({})>, affine_map<({}) -> ({})>]}} : ({}{}) -> {}\n\n", 
                        imm_tensor.mlir_id, tensor.mlir_id, dst_ids, tensor.format, imm_tensor.format, src_affine, src_affine, src_affine, dst_affine, src_dims, ", !ta.range".repeat(tr_indices.len()), dst_dims);
                    Ok(res)
                } else {
                    abort!(
                        self.span(),
                        "Transpose must be assigned to an intermediate tensor"
                    );
                }
            }
            CometUnaryOp::Sum => {
                if let Some(imm_scalar) = self.imm_expr.scalar() {
                    let res = format!(
                        "%{} = \"ta.reduce\"(%{}) : ({}) -> {}\n\n",
                        imm_scalar.mlir_id, tensor.mlir_id, src_dims, tensor.ty
                    );
                    Ok(res)
                } else {
                    abort!(
                        self.span(),
                        "Sum must be assigned to an intermediate scalar"
                    );
                }
            }
        }
    }
    fn scalar_emit_mlir(&self, _scalar: &ScalarStruct) -> Result<String> {
        Ok(String::new())
    }
}
