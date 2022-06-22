use crate::{cometexpr::*, CometVars};
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};
use syn::{Error, Ident};

#[derive(Debug, Clone)]
pub(crate) struct CometScalar {
    pub(crate) name: Ident,
    pub(crate) scalar: ScalarStruct,
}

impl CometDsl for CometScalar {
    fn emit_mlir(&mut self) -> Result<String> {
        // Ok(String::new()) //we don't emit anything for tensors(handled at declaration or as part of another expression)
        Ok(format!("{}", self.scalar.mlir_id))
    }
    fn my_parse(input: ParseStream, vars: &mut CometVars, _object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometScalar {input}");
        let fork = input.fork();
        if let Ok(scalar) = ScalarStruct::my_parse_no_semi(&fork, vars, _object_id) {
            //new scalar encountered
            input.advance_to(&fork);
            Ok(CometScalar {
                name: scalar.name.clone(),
                scalar: scalar,
            })
        } else {
            let name: Ident = fork.parse()?;
            if let Some(scalar) = vars.scalars.get(&name) {
                input.advance_to(&fork);
                Ok(CometScalar {
                    name,
                    scalar: scalar.clone(),
                })
            } else {
                // println!("Scalar not found");
                Err(Error::new(name.span(), "Scalar not found"))
            }
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        None
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        None
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        Some(&mut self.scalar)
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        Some(&self.scalar)
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.scalar.mlir_id = id;
    }
    fn span(&self) -> Span {
        self.name.span()
    }
}
