use crate::{cometexpr::*, CometVars};
use proc_macro_error::{abort_if_dirty, emit_error};
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};

#[derive(Debug, Clone)]
pub(crate) struct CometAssign {
    pub(crate) left: Box<CometExpr>,
    pub(crate) right: Box<CometExpr>,
}

impl CometDsl for CometAssign {
    fn emit_mlir(&mut self) -> Result<String> {
        let mut res = self.right.emit_mlir()?;
        if let Some(left) = self.left.tensor() {
            if left.format == TensorFormat::Dense && left.fill == TensorFill::None {
                emit_error!(
                    left.name.span(),
                    "Cannot assign to a dense tensor without a fill"
                );
                emit_error!(
                    self.right.span(),
                    "Cannot assign to a dense tensor without a fill"
                );
                abort_if_dirty();
                // return Err(Error::new(fork.span(), "Cannot assign to a dense tensor without a fill"));
            }
            if let Some(right) = self.right.tensor() {
                let (_, left_dims) = left.index_ids_and_dims();
                let (_, right_dims) = right.index_ids_and_dims();
                // res = format! {"{}\"ta.set_op\"(%{}, %{}) {{__beta__ = 0.000000e+00 : {}}} : ({}, {}) -> ()\n\n",res,right.mlir_id,left.mlir_id,left.ty,right_dims,left_dims};
                res = format! {"{}\"ta.set_op\"(%{}, %{}) : ({}, {}) -> ()\n\n",res,right.mlir_id,left.mlir_id,right_dims,left_dims};

            }
        }
        Ok(res)
    }
    fn my_parse(input: ParseStream, vars: &mut CometVars, object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometAssign {input}");
        let fork = input.fork();
        match CometExpr::my_parse(&fork, vars, object_id) {
            Ok(res) => {
                input.advance_to(&fork);
                Ok(CometAssign {
                    left: Box::new(CometExpr::None),
                    right: Box::new(res),
                })
            }
            Err(e) => {
                // println!("parse assign error {:?}", e);
                return Err(e);
            }
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        self.left.tensor_mut()
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        self.left.tensor()
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        self.left.scalar_mut()
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        self.left.scalar()
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.right.set_mlir_id(id);
    }
    fn span(&self) -> Span {
        self.right.span()
    }
}
