use crate::CometVars;
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};
use syn::Ident;
use syn::Token;

use proc_macro2::Span;

#[derive(Debug, Clone)]
pub(crate) struct ScalarStruct {
    pub(crate) name: Ident,
    pub(crate) ty: Ident,
    pub(crate) _mutable: bool,
    pub(crate) mlir_id: usize,
    // pub(crate) value: Box<CometExpr>,
}

impl ScalarStruct {
    pub(crate) fn emit_mlir(&mut self) -> String {
        String::new()
    }
    pub(crate) fn my_parse(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
        
    ) -> Result<Self> {
        // println!("parse ScalarStruct {input}");
        let fork = input.fork();
        fork.parse::<Token![let]>()?;
        let mut _mutable = false;
        if let Ok(_) = fork.parse::<Token![mut]>() {
            _mutable = true;
        }
        let name = fork.parse::<Ident>()?;
        let mut ty = Ident::new("f64", Span::call_site());
        if let Ok(_) = fork.parse::<Token![:]>() {
            ty = fork.parse::<Ident>()?;
        }
        fork.parse::<Token![;]>()?;

        let res = ScalarStruct {
            name,
            ty: ty,
            _mutable,
            mlir_id: *object_id,
        };
        *object_id += 1;
        input.advance_to(&fork);
        vars.scalars
            .entry(res.name.clone())
            .and_modify(|e| *e = res.clone())
            .or_insert(res.clone());
        Ok(res)
    }
    pub(crate) fn my_parse_no_semi(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
        
    ) -> Result<Self> {
        // println!("parse no semi ScalarStruct {input}");
        let fork = input.fork();
        fork.parse::<Token![let]>()?;
        let mut _mutable = false;
        if let Ok(_) = fork.parse::<Token![mut]>() {
            _mutable = true;
        }
        let name = fork.parse::<Ident>()?;
        let mut ty = Ident::new("f64", Span::call_site());
        if let Ok(_) = fork.parse::<Token![:]>() {
            ty = fork.parse::<Ident>()?;
        }

        let res = ScalarStruct {
            name,
            ty: ty,
            _mutable,
            mlir_id: *object_id,
        };
        *object_id += 1;
        input.advance_to(&fork);
        vars.scalars
            .entry(res.name.clone())
            .and_modify(|e| *e = res.clone())
            .or_insert(res.clone());
        Ok(res)
    }
}
