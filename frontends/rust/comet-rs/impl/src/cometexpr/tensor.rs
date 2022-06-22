use crate::{cometexpr::*, CometVars};
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};
use syn::{Error, Ident};

#[derive(Debug, Clone)]
pub(crate) struct CometTensor {
    pub(crate) name: Ident,
    pub(crate) tensor: TensorStruct,
    // pub(crate) indices: IndicesList,
}

impl CometDsl for CometTensor {
    fn emit_mlir(&mut self) -> Result<String> {
        // Ok(String::new()) //we don't emit anything for tensors(handled at declaration or as part of another expression)
        Ok(format!("{}", self.tensor.mlir_id))
    }
    fn my_parse(input: ParseStream, vars: &mut CometVars, _object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometTensor {input}");
        let fork = input.fork();
        let name: Ident = fork.parse()?;
        if let Some(tensor) = vars.tensors.get(&name) {
            input.advance_to(&fork);
            Ok(CometTensor {
                name,
                tensor: tensor.clone(),
            })
        } else {
            // println!("Tensor not found");
            Err(Error::new(name.span(), "Tensor not found"))
        }
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        Some(&mut self.tensor)
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        Some(&self.tensor)
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        None
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        None
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.tensor.mlir_id = id;
    }
    fn span(&self) -> Span {
        self.name.span()
    }
}

impl CometTensor{
    pub(crate) fn new_intermediate(tensor: &CometTensor, vars: &mut CometVars, object_id: &mut usize) -> CometTensor {
        let mut imm = tensor.clone();
        imm.tensor.name = format_ident!("imm_tensor_{}",*object_id);
        imm.tensor.fill = TensorFill::None;
        imm.tensor.mlir_id = *object_id;
        *object_id += 1;
        vars.tensors
                    .entry(imm.tensor.name.clone())
                    .and_modify(|e| *e = imm.tensor.clone())
                    .or_insert(imm.tensor.clone());
        imm
    }
}
