use proc_macro2::Span;
use proc_macro_error::abort;
use std::collections::{HashMap, HashSet};
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};
use syn::{bracketed, parenthesized, Token};
use syn::{Error, Ident, LitInt, Type};

use crate::CometVars;

#[derive(Debug, Clone)]
pub(crate) struct IndicesList {
    pub indices: Vec<IndexStruct>,
    pub spans: HashMap<IndexStruct, Span>,
}

impl std::cmp::PartialEq for IndicesList {
    fn eq(&self, other: &Self) -> bool {
        self.indices == other.indices
    }
}

impl IndicesList {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &IndexStruct> + DoubleEndedIterator {
        self.indices.iter()
    }
    pub(crate) fn union(&self, other: &IndicesList) -> Self {
        let mut indices = HashSet::new();
        let mut spans = HashMap::new();
        for i in self.indices.iter() {
            indices.insert(i.clone());
            spans.insert(i.clone(), self.spans[i].clone());
        }
        for i in other.indices.iter() {
            indices.insert(i.clone());
            spans.insert(i.clone(), other.spans[i].clone());
        }
        let mut indices_list = indices.into_iter().collect::<Vec<IndexStruct>>();
        indices_list.sort();
        IndicesList {
            indices: indices_list,
            spans: spans,
        }
    }
    pub(crate) fn contains(&self, index: &IndexStruct) -> bool {
        if let Some(_) = self.spans.get(index) {
            //faster than linear search through vector
            true
        } else {
            false
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }
    pub(crate) fn as_affine(&self) -> String {
        self.indices
            .iter()
            .fold(String::new(), |acc, x| format!("{} d{},", acc, x.mlir_id))
            .trim_end_matches(',')
            .to_string()
    }
    pub(crate) fn span(&self, index: &IndexStruct) -> Span {
        self.spans[index]
    }

    pub(crate) fn my_parse(input: ParseStream, vars: &mut CometVars) -> Result<Self> {
        if !input.peek(syn::token::Bracket) {
            abort!(
                input.span(),
                "Expected index list... e.g. Tensor::<_>::dense([a,b,c])"
            );
        }
        let indices;
        bracketed!(indices in input);
        let mut indices_list = Vec::new();
        let mut spans = HashMap::new();
        while !indices.is_empty() {
            match indices.parse::<Ident>() {
                Ok(ident) => {
                    if let Some(index) = vars.indices.get(&ident) {
                        indices_list.push(index.clone());
                        spans.insert(index.clone(), ident.span());
                    } else {
                        abort!(ident.span(), "Unknown index {}", ident);
                    }
                }
                Err(_) => {
                    abort!(
                        indices.span(),
                        "2. Expected index list... e.g. Tensor::<_>::dense([a,b,c])"
                    );
                }
            }
            if indices.peek(Token![,]) {
                indices.parse::<Token![,]>()?;
            }
        }
        Ok(IndicesList {
            indices: indices_list,
            spans: spans,
        })
    }
}

impl std::ops::Index<usize> for IndicesList {
    type Output = IndexStruct;
    fn index(&self, index: usize) -> &Self::Output {
        &self.indices[index]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndexValue {
    Constant(LitInt),
    Dynamic,
}

impl IndexValue {
    pub fn emit_mlir(&self) -> String {
        match self {
            IndexValue::Constant(val) => val.to_string(),
            IndexValue::Dynamic => "?".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct IndexStruct {
    pub(crate) name: Ident,
    value: IndexValue,
    pub(crate) mlir_id: usize,
}

impl std::cmp::Ord for IndexStruct {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name.cmp(&other.name)
    }
}

impl std::cmp::PartialOrd for IndexStruct {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl IndexStruct {
    pub(crate) fn emit_mlir(&self) -> String {
        let mut res = "".to_string();
        res = format!("{}%c{}{} = arith.constant 0: index\n", res, self.mlir_id, 0);
        res = format!("{}%c{}{} = arith.constant 1: index\n", res, self.mlir_id, 1);
        if let Ok(val) = self.val() {
            res = format!(
                "{}%c{}{} = arith.constant {}: index\n",
                res, self.mlir_id, val, val
            );
            format!("{}%{} = \"ta.static_index_label\"(%c{}{}, %c{}{}, %c{}{}) : (index, index, index) -> !ta.range\n\n",res, self.mlir_id, self.mlir_id, 0, self.mlir_id, val, self.mlir_id,1)
        } else {
            format!("{}%{} = \"ta.dynamic_index_label\"(%c{}{}, %c{}{}) : (index, index) -> !ta.range\n\n",
                    res, self.mlir_id, self.mlir_id, 0, self.mlir_id,1)
        }
    }

    pub(crate) fn val(&self) -> Result<i32> {
        match &self.value {
            IndexValue::Constant(val) => val.base10_parse(),
            IndexValue::Dynamic => {
                Err(Error::new(Span::call_site(), "Index value is not constant"))
            }
        }
    }

    pub(crate) fn val_mlir(&self) -> String {
        self.value.emit_mlir()
    }

    pub(crate) fn my_parse(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Vec<Self>> {
        let fork = input.fork();
        // println!("object_id {:?}", object_id);
        let mut index = IndexStruct {
            name: Ident::new("temp", Span::call_site()),
            value: IndexValue::Dynamic,
            mlir_id: *object_id,
        };
        
        fork.parse::<Token![let]>()?;

        let mut idents = vec![];

        while !fork.peek(Token![=]) {
            idents.push(fork.parse::<Ident>()?);

            if let Ok(_) = fork.parse::<Token![:]>() {
                fork.parse::<Type>()?; //TODO ensure type matches Index
            }
            if fork.peek(Token![,]) {
                fork.parse::<Token![,]>()?;
            }
        }
       
        fork.parse::<Token![=]>()?;

        let mut valid = false;
        while let Ok(ident) = fork.parse::<Ident>() {
            if ident == "Index" {
                valid = true;
            } else if ident == "new" {
                index.value = IndexValue::Dynamic;
                let _content;
                parenthesized!(_content in fork);
                break;
            } else if ident == "with_value" {
                let content;
                parenthesized!(content in fork);
                match content.parse::<LitInt>() {
                    Ok(value) => index.value = IndexValue::Constant(value),
                    Err(_err) => {
                        abort!(ident, "Index value must be a positive integer");
                    }
                }
                break;
            }
            fork.parse::<Token![::]>()?;
        }
        fork.parse::<Token![;]>()?;

        vars.indices
            .entry(index.name.clone())
            .and_modify(|e| *e = index.clone())
            .or_insert(index.clone());
        // println!("object_id {:?}", object_id);
        // println!("{valid} {:#?}", index);
        let mut indices = vec![];
        if valid {
            for name in idents{
                let mut new_index = index.clone();
                new_index.name = name;
                new_index.mlir_id = *object_id;
                *object_id += 1;
                vars.indices
                    .entry(new_index.name.clone())
                    .and_modify(|e| *e = new_index.clone())
                    .or_insert(new_index.clone());
                indices.push(new_index);
            }
            input.advance_to(&fork);
            Ok(indices)
        } else {
            Err(Error::new(input.span(), "no index found"))
        }
    }
}
