use crate::index::IndicesList;
use crate::CometVars;

use proc_macro2::Span;
use proc_macro_error::abort;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::{parenthesized, Token};
use syn::{Error, Ident, Lit, LitStr, LitInt};

#[derive(Debug, Clone)]
pub(crate) struct TensorStruct {
    pub(crate) name: Ident,
    pub(crate) ty: Ident, //we could store the Type but im not sure it actually buys us anything, and would be more complicated to work with
    pub(crate) format: TensorFormat,
    pub(crate) temporal: bool,
    pub(crate) indices: IndicesList,
    pub(crate) fill: TensorFill,
    pub(crate) mutable: bool,
    pub(crate) mlir_id: usize,
    // value: Option<Lit>
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GenericFormat {
    CU,
    CN,
    D,
    S,
}

impl std::fmt::Display for GenericFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericFormat::CU => write!(f, "CU"),
            GenericFormat::CN => write!(f, "CN"),
            GenericFormat::D => write!(f, "D"),
            GenericFormat::S => write!(f, "S"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TensorFormat {
    Dense,
    Csr,
    Dcsr,
    Coo,
    Csf,
    Generic(Vec<GenericFormat>),
}

impl Parse for TensorFormat {
    fn parse(input: ParseStream) -> Result<Self> {
        let temp: syn::Ident = input.parse()?;
        let tstr = temp.to_string();
        if tstr == "generic" {
            if let Ok(generic_formats) = input.parse::<syn::AngleBracketedGenericArguments>() {
                let mut generic_formats_vec = Vec::new();
                for generic_format in generic_formats.args.iter() {
                    match generic_format {
                        syn::GenericArgument::Type(ty) => match ty {
                            syn::Type::Path(syn::TypePath {
                                qself: None,
                                path: syn::Path { segments, .. },
                            }) => {
                                for segment in segments.iter() {
                                    match segment {
                                        syn::PathSegment { ident, .. } => {
                                            let ident_str = ident.to_string();
                                            if ident_str == "CU" {
                                                generic_formats_vec.push(GenericFormat::CU);
                                            } else if ident_str == "CN" {
                                                generic_formats_vec.push(GenericFormat::CN);
                                            } else if ident_str == "D" {
                                                generic_formats_vec.push(GenericFormat::D);
                                            } else if ident_str == "S" {
                                                generic_formats_vec.push(GenericFormat::S);
                                            } else {
                                                abort!(
                                                    ident.span(),
                                                    "Unknown generic format {}",
                                                    ident_str
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                abort!(generic_format.span(), "Unknown generic format");
                            }
                        },
                        _ => {
                            abort!(generic_formats.span(), "Unknown generic format");
                        }
                    }
                }
                Ok(TensorFormat::Generic(generic_formats_vec))
            } else {
                abort!(input.span(), "generic format must specify a list of generic formats e.g. tensor::<f64>::generic<CU,CN,D,S>([...])");
            }
        } else {
            Ok(TensorFormat::from(tstr.as_str()))
        }
    }
}

impl std::fmt::Display for TensorFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorFormat::Dense => write!(f, "Dense"),
            TensorFormat::Csr => write!(f, "CSR"),
            TensorFormat::Dcsr => write!(f, "DCSR"),
            TensorFormat::Coo => write!(f, "COO"),
            TensorFormat::Csf => write!(f, "CSF"),
            TensorFormat::Generic(g) => {
                for i in 0..g.len() - 1 {
                    write!(f, "{}, ", g[i])?;
                }
                write!(f, "{}", g[g.len() - 1])
            }
        }
    }
}

// impl ToTokens for TensorFormat {
//     fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
//         let ident = Ident::new(&self.to_string(), Span::call_site());
//         ident.to_tokens(tokens);
//     }
// }


impl From<&str> for TensorFormat {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "dense" => TensorFormat::Dense,
            "csr" => TensorFormat::Csr,
            "dcsr" => TensorFormat::Dcsr,
            "coo" => TensorFormat::Coo,
            "csf" => TensorFormat::Csf,
            "generic" => TensorFormat::Generic(vec![
                GenericFormat::CU,
                GenericFormat::CN,
                GenericFormat::D,
                GenericFormat::S,
            ]),
            _ => abort!(Span::call_site(), "Unknown tensor format: {}", s),
        }
    }
}


impl TensorStruct {
    pub(crate) fn index_ids_and_dims(&self) -> (String, String) {
        let (ids, dims) = self
            .indices
            .iter()
            .fold((String::new(), String::new()), |acc, idx| {
                (
                    format!("{} %{},", acc.0, idx.mlir_id),
                    format!("{}{}x", acc.1, idx.val_mlir()),
                )
            });
        let ids = ids.strip_suffix(",").expect("trying to strip commas");
        let dims = format!("tensor<{}{}>", dims, self.ty.to_string());
        (ids.to_string(), dims)
    }
    pub(crate) fn emit_mlir(&self) -> String {
        let (ids, dims) = self.index_ids_and_dims();
        let mut ta_range = ", ".to_string();
        let mut ids = self.indices.iter().fold(
            String::new(), |acc, idx| {
                if idx.val_mlir() == "?"  {
                    match idx.dim_of {
                        Some((tid, dim))  =>    {
                            if tid != self.mlir_id {
                                format!("{} %d{}_{},", acc, tid, dim)
                            }
                            else {
                                acc
                            }
                        }
                        None => {
                            acc
                        }
                    }
                }
                else {
                    acc
                }
            }
        );
        if ids != "" {
            ids = ids.strip_suffix(",").expect("trying to strip commas").to_string();
            ta_range = "index, ".repeat(ids.split(',').collect::<Vec<_>>().len()).to_string();
        }
        let res;
        match self.format {
            TensorFormat::Dense => {
                res = format!(
                    "%{} = \"ta.dense_tensor_decl\"({}) {{format = \"{}\"}} : ({}) -> {}\n",
                    self.mlir_id,
                    ids,
                    self.format,
                    &ta_range[0..ta_range.len() - 2],
                    dims
                );
            }
            _ => {
                let mut temp_res = format!(
                    "%{} = \"ta.sparse_tensor_decl\"({}) {{format = \"{}\", temporal_tensor = {}}} : ({}) -> {}\n",
                    self.mlir_id,
                    ids,
                    self.format,
                    self.temporal,
                    &ta_range[0..ta_range.len() - 2],
                    dims
                );
                let dims_len = dims.split('x').count() -1; 
                for i in 0..dims_len {
                    if self.indices[i].dim_of.unwrap().0 == self.mlir_id {

                        temp_res = format!(
                            "{}%c{1}_{2} = arith.constant {2}: index\n", temp_res, self.mlir_id, i
                        );
                        temp_res = format!(
                            "{0}%d{1}_{2} = \"ta.dim\"(%{1}, %c{1}_{2}) : ({3},index) -> index \n", temp_res, self.mlir_id, i, dims 
                        );
                    }
                }
                res = temp_res
            }
        }
        if self.fill != TensorFill::None {
            format!(
                "{}{} : ({}) -> ()\n\n",
                res,
                self.fill.emit_mlir(self.mlir_id, &self.ty),
                dims
            )
        } else {
            format!("{}\n", res)
        }
    }

    pub(crate) fn my_parse(
        input: ParseStream,
        vars: &mut CometVars,
        object_id: &mut usize,
    ) -> Result<Self> {
        let fork = input.fork();
        fork.parse::<Token![let]>()?;
        let mut mutable = false;
        if let Ok(_) = fork.parse::<Token![mut]>() {
            //TODO: handle mut
            mutable = true;
        }
        let name = fork.parse::<Ident>()?;
        let mut ty: Option<Ident> = None;
        if let Ok(_) = fork.parse::<Token![:]>() {
            ty = Some(fork.parse::<Ident>()?);
        }
        fork.parse::<Token![=]>()?;

        let mut valid = false;
        while !valid {
            //either we parse correctly or we run out of things to parse and return error
            let ident = fork.parse::<Ident>()?;
            if ident == "Tensor" {
                valid = true;
                if fork.peek3(Token![<]) {
                    fork.parse::<Token![::]>()?;
                    fork.parse::<Token![<]>()?;
                    ty = Some(fork.parse::<Ident>()?);
                    fork.parse::<Token![>]>()?;
                }
                fork.parse::<Token![::]>()?;
            }
        }
        let format = match fork.parse::<TensorFormat>() {
            Ok(format) => format,
            Err(_) => abort!(fork.span(), "invalid format"),
        };

        let content;
        parenthesized!(content in fork);
        let mut indices = IndicesList::my_parse(&content, vars)?; //.parse::<IndicesList>()?;
        if let TensorFormat::Generic(g) = &format {
            if g.len() != indices.len() {
                abort!(
                    fork.span(),
                    "Generic format must have the same number of indices as the tensor"
                );
            }
        }

        for (dim,index) in indices.indices.iter_mut().enumerate() {
            if let Some((_,_))  = index.dim_of {

            }
            else {
                index.dim_of = Some((*object_id, dim));
                if let Some(var) = vars.indices.get_mut(&index.name) {

                    var.dim_of = Some((*object_id, dim));       
                }
                else{

                    panic!("Could not find index: {}", index.name);
                }
            }
        }

        let mut fill = if fork.peek(Token![.]) {
            fork.parse::<Token![.]>()?;
            fork.parse::<TensorFill>()?
        } else {
            TensorFill::None
        };
        if let Err(_) = fork.parse::<Token![;]>(){
            abort!(fork.span(), "expected ; after tensor declaration");
        
        }
        let ty = match ty {
            Some(ty) => ty,
            None => abort!(fork.span(), "No type specified"),
        };

        match & mut fill {
            TensorFill::None => (),
            TensorFill::Fill(f) => {
                if format != TensorFormat::Dense {
                    abort!(f.span(), "sparse tensors cannot be filled from a constant");
                }
            }
            TensorFill::FillFromFile(val,env,_) => {
                if format == TensorFormat::Dense {
                    abort!(val.span(), "Dense tensors cannot be filled from file");
                }
                *env = format!("SPARSE_FILE_NAME{}",*object_id); //uncomment this when we get the proper API
                // *env = format!("SPARSE_FILE_NAME");
                std::env::set_var(env,val.value());
            }
        }
        // println!("object_id {:?}", object_id);
        let tensor = TensorStruct {
            name: name,
            ty: ty,
            format: format,
            temporal: false,
            indices: indices,
            fill: fill,
            mutable: mutable,
            mlir_id: *object_id,
        };
        *object_id += 1;
        vars.tensors
            .entry(tensor.name.clone())
            .and_modify(|e| *e = tensor.clone())
            .or_insert(tensor.clone());
        // println!("object_id {:?}", object_id);
        input.advance_to(&fork);
        Ok(tensor)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TensorFill {
    Fill(Lit),
    FillFromFile(LitStr,String,LitInt),
    None,
}

impl TensorFill {
    pub(crate) fn emit_mlir(&self, mlir_id: usize, ty: &Ident) -> String {
        let res = "\"ta.".to_string();
        match self {
            TensorFill::Fill(lit) => {
                let val = match lit {
                    Lit::Int(lit) => lit.to_string(),
                    Lit::Float(lit) => lit.to_string(),
                    _ => {
                        abort!(lit.span(), "Only int and float literals are supported");
                    }
                };
                format!(
                    "{}fill\"(%{}) {{value = {} : {} }}",
                    res,
                    mlir_id,
                    val,
                    ty.to_string()
                )
            }
            TensorFill::FillFromFile(_,envar,mode) => {
                format!(
                    "{}fill_from_file\"(%{}) {{filename = \"{}\", readMode = {} : i32}}",
                    res,
                    mlir_id,
                    envar,
                    mode.to_string(),
                )
            }
            TensorFill::None => {
                format! {""}
                // format!("{}fill\"(%{}) {{}} : ", res, mlir_id, ty.to_string())
            }
        }
    }
    pub(crate) fn span(&self) -> Span {
        match self {
            TensorFill::Fill(lit) => lit.span(),
            TensorFill::FillFromFile(litstr,_,_) => litstr.span(),
            TensorFill::None => Span::call_site(),
        }
    }
}

impl Parse for TensorFill {
    fn parse(input: ParseStream) -> Result<Self> {
        let fork = input.fork();
        match fork.parse::<Ident>()?.to_string().as_str() {
            "fill" => {
                let content;
                parenthesized!(content in fork);
                input.advance_to(&fork);
                Ok(TensorFill::Fill(content.parse::<Lit>()?))
            }
            "load" => {
                let content;
                parenthesized!(content in fork);
                input.advance_to(&fork);
                let file = content.parse::<LitStr>()?;
                let mode = if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                    content.parse::<LitInt>()?
                }
                else {
                    syn::LitInt::new("1",file.span())
                };
                Ok(TensorFill::FillFromFile(file,String::new(),mode))
            }
            _ => Err(Error::new(input.span(), "invalid fill")),
        }
    }
}

