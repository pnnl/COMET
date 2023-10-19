use crate::{cometexpr::*, CometVars};
use proc_macro_error::abort;
use syn::parse::discouraged::Speculative;
use syn::parse::{ParseStream, Result};
use syn::Token;

#[derive(Debug, Clone)]
pub(crate) enum SemiringOp{
    Min,
    Plus,
    Times,
    Any,
    First,
    Second,
    Pair,
}

impl SemiringOp {
    fn emit_mlir(&self) -> String {
        match self {
            SemiringOp::Min => "minxy".to_string(),
            SemiringOp::Plus => "plusxy".to_string(),
            SemiringOp::Times => "times".to_string(),
            SemiringOp::Any => "any".to_string(),
            SemiringOp::First => "first".to_string(),
            SemiringOp::Second => "second".to_string(),
            SemiringOp::Pair => "pairxy".to_string(),
        }
    }
    fn parse_str(input: &str) -> Result<Vec<Self>> {
        let mut ops = Vec::new();
        let str_ops = input.split(",");
        for op in str_ops {
            match op.trim() {
                "min" => ops.push(SemiringOp::Min),
                "+" => ops.push(SemiringOp::Plus),
                "*" => ops.push(SemiringOp::Times),
                "any" => ops.push(SemiringOp::Any),
                "first" => ops.push(SemiringOp::First),
                "second" => ops.push(SemiringOp::Second),
                "pair" => ops.push(SemiringOp::Pair),
                _ => {
                    abort!(
                        Span::call_site(),
                        "Unsupported semiring operation: {}",
                        op
                    );
                }
            }
        }
        Ok(ops)
    }
}

impl std::fmt::Display for SemiringOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.emit_mlir())
    }
}

#[derive(Debug, Clone)]
pub(crate) enum CometBinOp {
    Mul,
    EleWiseMul,
    Semiring(Vec<SemiringOp>),
}

#[derive(Debug, Clone)]
pub(crate) struct CometBinary {
    pub root: Box<CometExpr>,
    pub left: Box<CometExpr>,
    pub op: CometBinOp,
    pub right: Box<CometExpr>,
}

impl CometBinOp {
    pub(crate) fn my_parse(input: ParseStream, vars: &mut CometVars) -> Result<Self> {
        // println!("parse CometBinOp {input}");
        if input.peek(Token![*]) {
            input.parse::<Token![*]>()?;
            Ok(CometBinOp::Mul)
        } else if input.peek(Token![|]) {
            //need to make sure we never use this operator
            input.parse::<Token![|]>()?;
            if let Some(op) = vars.custom_ops.get("|"){
                if ".*" == op{
                    Ok(CometBinOp::EleWiseMul)
                }
                else{
                    abort!(
                        input.span(),
                        "Unsupported operation: {}",
                        op
                    );
                }
            }
            else{
                abort!(
                    input.span(),
                    "Unsupported operation"
                );
            }
        }
        else if input.peek(Token![&]){ 
            input.parse::<Token![&]>()?;
            if let Some(op) = vars.custom_ops.get("&"){
                if let Ok(ops) = SemiringOp::parse_str(op){
                    Ok(CometBinOp::Semiring(ops))
                }
                else{
                    abort!(
                        Span::call_site(),
                        "Unsupported semiring operation: {}",
                        op
                    );
                }
            }
            else{
                abort!(
                    input.span(),
                    "Unsupported operation:"
                );
            }
        } else {
            abort!(input.span(), "Unsupported operation");
        }
    }
}

impl CometDsl for CometBinary {
    fn emit_mlir(&mut self) -> Result<String> {
        // println!("emit_mlir cometBinary");
        let left_str = self.left.emit_mlir()?;
        let right_str = self.right.emit_mlir()?;
        let mut res = format!("{}{}", left_str, right_str);
        if let Some(left) = self.left.tensor() {
            if let Some(right) = self.right.tensor() {
                res = format!("{}{}", res, self.tensor_tensor_emit_mlir(left, right)?);
            } else if let Some(right) = self.right.scalar() {
                res = format!("{}{}", res, self.tensor_scalar_emit_mlir(left, right)?);
            } else {
                abort!(self.root.span(), "Unsupported operation");
            }
        } else if let Some(left) = self.left.scalar() {
            if let Some(right) = self.left.tensor() {
                res = format!("{}{}", res, self.tensor_scalar_emit_mlir(right, left)?);
            }
            if let Some(right) = self.right.scalar() {
                res = format!("{}{}", res, self.scalar_scalar_emit_mlir(left, right)?);
            } else {
                abort!(self.root.span(), "Unsupported operation");
            }
        } else {
            abort!(self.root.span(), "Unsupported operation");
        }
        Ok(res)
    }
    fn my_parse(input: ParseStream, vars: &mut CometVars, object_id: &mut usize, ) -> Result<Self> {
        // println!("parse CometBinary {input}");
        let fork = input.fork();
        let root = Box::new(CometExpr::None); //this will be replaced by whoever calls this
        let left = Box::new(CometExpr::my_parse(&fork, vars, object_id)?);
        let op = CometBinOp::my_parse(&fork, vars)?;
        let right = Box::new(CometExpr::my_parse(&fork, vars, object_id)?);
        input.advance_to(&fork);
        Ok(CometBinary {
            root,
            left,
            op,
            right,
        })
    }
    fn tensor_mut(&mut self) -> Option<&mut TensorStruct> {
        self.root.tensor_mut()
    }
    fn tensor(&self) -> Option<&TensorStruct> {
        self.root.tensor()
    }
    fn scalar_mut(&mut self) -> Option<&mut ScalarStruct> {
        self.root.scalar_mut()
    }
    fn scalar(&self) -> Option<&ScalarStruct> {
        self.root.scalar()
    }
    fn set_mlir_id(&mut self, id: usize) {
        self.root.set_mlir_id(id);
    }
    fn span(&self) -> Span {
        self.root.span()
    }
}

impl CometBinary {
    pub(crate) fn assign_tensor_root(&mut self, root: CometTensor,vars: &mut CometVars, object_id: &mut usize) {
        self.root = Box::new(CometExpr::Tensor(CometTensor::new_intermediate(&root, vars, object_id)));
        if let None = self.left.tensor() { 
            if let CometExpr::Binary(ref mut bin) = *self.left {//TODO calulate the appropriate shape,indices to use
                let tensor = CometTensor::new_intermediate(&root, vars, object_id);
                bin.assign_tensor_root(tensor, vars, object_id);
            }
        }
        if let None = self.right.tensor() { 
            if let CometExpr::Binary(ref mut bin) = *self.left { //TODO calulate the appropriate shape,indices to use
                let tensor = CometTensor::new_intermediate(&root, vars, object_id);
                bin.assign_tensor_root(tensor, vars, object_id);
            }
        }
    }
    fn tensor_tensor_emit_mlir(&self, left: &TensorStruct, right: &TensorStruct) -> Result<String> {
        let (_, left_dims) = left.index_ids_and_dims();
        let left_affine = left.indices.as_affine();

        let (_, right_dims) = right.index_ids_and_dims();
        let right_affine = right.indices.as_affine();

        let data_type = left.ty.clone();
        if let Some(res_tensor) = self.root.tensor() {
            let (res_mlir_ids, res_dims) = res_tensor.index_ids_and_dims();
            let res_affine = res_tensor.indices.as_affine();
            match &self.op {
                CometBinOp::Mul => {                   
                    let union_ids = left.indices.union(&right.indices);
                    for id in res_tensor.indices.iter() {
                        if !union_ids.contains(&id) {
                            abort!(
                                res_tensor.indices.span(id),
                                "Output matrix index not found in either input matrix"
                            );
                        }
                    }
                    let affine = format!("{}", union_ids.as_affine());
                    let res = format!("%{} = \"ta.mul\"(%{},%{}, {}) {{MaskType = \"none\", __alpha__ = 1.000000e+00 : {}, __beta__ = 0.000000e+00 : {}, formats = [\"{}\", \"{}\", \"{}\"], indexing_maps = [affine_map<({}) -> ({})>, affine_map<({}) -> ({})>, affine_map<({}) -> ({})>], operand_segment_sizes = array<i32: 1, 1, {}, 0>, semiring = \"plusxy_times\"}} : ({}, {}{}) -> {}\n\n", 
                        res_tensor.mlir_id, left.mlir_id, right.mlir_id, res_mlir_ids, data_type, data_type, left.format, right.format, res_tensor.format, affine, left_affine, affine, right_affine , affine, res_affine,  res_dims.split("x").count() -1, left_dims, right_dims, ", !ta.range".repeat(res_tensor.indices.len()), res_dims);
                    Ok(res)
                    
                }
                CometBinOp::EleWiseMul => {
                    let affine = res_affine.clone();
                    let res = format!("%{} = \"ta.elews_mul\"(%{},%{}, {}) {{__alpha__ = 1.000000e+00 : {}, __beta__ = 0.000000e+00 : {}, formats = [\"{}\", \"{}\", \"{}\"], indexing_maps = [affine_map<({}) -> ({})>, affine_map<({}) -> ({})>, affine_map<({}) -> ({})>], semiring = \"noop_times\"}} : ({}, {}{}) -> {}\n\n", 
                        res_tensor.mlir_id, left.mlir_id, right.mlir_id, res_mlir_ids, data_type, data_type, left.format, right.format, res_tensor.format, affine, left_affine, affine, right_affine , affine, res_affine, left_dims, right_dims, ", !ta.range".repeat(res_tensor.indices.len()), res_dims);
                    Ok(res)
                    
                }
                CometBinOp::Semiring(ops) =>{
                    match ops.len(){
                        1 => {
                            let affine = res_affine.clone();
                            let res = format!("%{} = \"ta.elews_mul\"(%{},%{}, {}) {{__alpha__ = 1.000000e+00 : {}, __beta__ = 0.000000e+00 : {}, formats = [\"{}\", \"{}\", \"{}\"], indexing_maps = [affine_map<({}) -> ({})>, affine_map<({}) -> ({})>, affine_map<({}) -> ({})>], semiring = \"noop_{}\"}} : ({}, {}{}) -> {}\n\n", 
                                res_tensor.mlir_id, left.mlir_id, right.mlir_id, res_mlir_ids, data_type, data_type, left.format, right.format, res_tensor.format, affine, left_affine, affine, right_affine , affine, res_affine, ops[0], left_dims, right_dims, ", !ta.range".repeat(res_tensor.indices.len()), res_dims);
                            Ok(res)
                        }, //elews_mul
                        2 => {
                            let union_ids = left.indices.union(&right.indices);
                            for id in res_tensor.indices.iter() {
                                if !union_ids.contains(&id) {
                                    abort!(
                                        res_tensor.indices.span(id),
                                        "Output matrix index not found in either input matrix"
                                    );
                                }
                            }
                            let affine = format!("{}", union_ids.as_affine());
                            let res = format!("%{} = \"ta.mul\"(%{},%{}, {}) {{MaskType = \"none\", __alpha__ = 1.000000e+00 : {}, __beta__ = 0.000000e+00 : {}, formats = [\"{}\", \"{}\", \"{}\"], indexing_maps = [affine_map<({}) -> ({})>, affine_map<({}) -> ({})>, affine_map<({}) -> ({})>], operand_segment_sizes = array<i32: 1, 1, {}, 0>, semiring = \"{}_{}\"}} : ({}, {}{}) -> {}\n\n", 
                                res_tensor.mlir_id, left.mlir_id, right.mlir_id, res_mlir_ids, data_type, data_type, left.format, right.format, res_tensor.format, affine, left_affine, affine, right_affine , affine, res_affine,  res_dims.split("x").count() -1, ops[0], ops[1], left_dims, right_dims, ", !ta.range".repeat(res_tensor.indices.len()), res_dims);
                            Ok(res)
                        }, //tc
                        _ => {
                            abort!(
                                self.root.span(),
                                "unsupported semiring operation"
                            );
                        }
                    }
                }
                // _ => {
                //     abort!(
                //         self.root.span(),
                //         "unsupported operation for tensor-tensor operation"
                //     );
                // }
            }
        }
        else {
            abort!(
                self.root.span(),
                "result of a tensor multiplication must be a tensor"
            );
        }
    }
    fn tensor_scalar_emit_mlir(
        &self,
        _left: &TensorStruct,
        _right: &ScalarStruct,
    ) -> Result<String> {
        Ok(format!("{}", "TODO"))
    }
    fn scalar_scalar_emit_mlir(
        &self,
        _left: &ScalarStruct,
        _right: &ScalarStruct,
    ) -> Result<String> {
        Ok(format!("{}", "TODO"))
    }
}
