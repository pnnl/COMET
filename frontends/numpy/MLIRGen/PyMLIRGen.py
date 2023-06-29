#
# Copyright 2022 Battelle Memorial Institute
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

from ast import operator
import jinja2
import functools
import itertools

from typing import Dict, List, Tuple, Sequence, Union
from collections import OrderedDict

import numpy

from MLIRGen import types_mlir


class Dialect:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name} dialect"

class Tensor_Decl_Builder:

    indentation_size = 4

    tensor_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.dense_tensor_decl"{{dims_tuple}}'
        + '{format = "Dense"} : '
        +"{{ranges_tuple}} -> "
        + "{{inputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, decl_vars:list, inputtype: str)->None:
        self.inputtype = inputtype
        self.decl_vars = decl_vars


    def build_tensor(self):
        dims_tuple = "("
        ranges_tuple = "("
        for i in range(len(self.decl_vars)-1):
            dims_tuple += self.decl_vars[i] + ","
            ranges_tuple += "!ta.range,"

        dims_tuple += self.decl_vars[-1] +  ")"
        ranges_tuple += "!ta.range)"

        return self.tensor_decl_wrapper_text.render(
            dims_tuple = dims_tuple,
            ranges_tuple = ranges_tuple,
            inputtype = self.inputtype

        )



class TC_and_TrPose_Builder:
    indentation_size = 4
    beta_val = 0.0

    tc_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.mul"{{operators}}'
        + '{__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = {{beta}} : f64,"
        + 'formats = ["Dense", "Dense", "Dense"],'
        +'indexing_maps = {{indexing_maps}}, semiring = "plusxy_times"} : '
        +"{{types_range_str}}"
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    elewisemult_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.elews_mul"{{operators}}'
        + '{__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = {{beta}} : f64,"
        + 'formats = ["Dense", "Dense", "Dense"],'
        +'indexing_maps = {{indexing_maps}}, semiring = "noop_times"} : '
        +"{{types_range_str}}"
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    tranpose_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.transpose"{{operators}}'
        + '{__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = {{beta}} : f64,"
        + 'formats = ["Dense", "Dense", "Dense"],'
        +'indexing_maps = {{indexing_maps}},semiring = "plusxy_times"} : '
        +"{{types_range_str}}"
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, input_tensors:list, dimslbls_to_map:list, input_array_dims_lbls:list, 
                            target_dims_lbls:list,tensor_types:list,tc_indices:list,opr_type:str,op:str) -> None:
        self.operators = input_tensors
        self.dimslbls_to_map = dimslbls_to_map
        self.input_array_dims_lbls = input_array_dims_lbls
        self.target_dims_lbls = target_dims_lbls
        self.tensor_types = tensor_types
        self.tc_indices = tc_indices
        self.opr_type = opr_type
        self.op = op

    def build_tc(self):

        out_tensor_type = self.tensor_types.pop()
        types_range_str = "("
        for type in self.tensor_types:
            types_range_str += type + ","

        for i in range(len(self.tc_indices)):

            self.operators.append(self.tc_indices[i])
            if(i < len(self.tc_indices)-1):
                types_range_str += "!ta.range,"
        types_range_str += "!ta.range)"

        self.operators = str(tuple(self.operators)).replace("'", "")
        
        indexing_maps = TC_and_TrPose_Builder.create_affine_mapping(self.dimslbls_to_map,self.input_array_dims_lbls,self.target_dims_lbls)

        self.beta_val = TC_and_TrPose_Builder.get_beta_val(self.op)        

        if(self.opr_type == "contraction"):
            return self.tc_decl_wrapper_text.render(
                operators = self.operators,
                indexing_maps = indexing_maps,
                types_range_str = types_range_str,
                outputtype = out_tensor_type,
                beta =  self.beta_val
            )
        elif(self.opr_type == "elewise_mult"):
            return self.elewisemult_wrapper_text.render(
                operators = self.operators,
                indexing_maps = indexing_maps,
                types_range_str = types_range_str,
                outputtype = out_tensor_type,
                beta =  self.beta_val
            )
        else:
            return self.tranpose_wrapper_text.render(
                operators = self.operators,
                indexing_maps = indexing_maps,
                types_range_str = types_range_str,
                outputtype = out_tensor_type,
                beta =  self.beta_val
            )

    def get_beta_val(op):
        if(op == '='):
            beta_val = '0.000000e+00'
        elif(op == '+='):
            beta_val = '1.000000e+00'
        elif(op == '-='):
            beta_val = '-1.000000e+00'

        return beta_val

    
    def create_affine_mapping(dims_to_map:list, input_array_dims:list, target_dims:list):
        
        input_map = {}
        for i in range(len(dims_to_map)):
            input_map[dims_to_map[i]] = "d{}".format(i)
        
        mapping_String = "["
        for list_dims in input_array_dims:
            d_map = []
            for dim in list_dims:
                d_map.append(input_map[dim])

            if(len(d_map) == 1):
                d_map = d_map.pop()
                mapping_String += "affine_map<" + str(tuple(input_map.values())).replace("'", "") + "-> ({})".format(d_map.replace("'", "")) + ">,"
            else:
                mapping_String += "affine_map<" + str(tuple(input_map.values())).replace("'", "") + "->" + str(tuple(d_map)).replace("'", "") + ">,"

        d_map = []
        for dim in target_dims:
            d_map.append(input_map[dim])


        if(len(d_map) == 1):
            d_map = d_map.pop()
            mapping_String += "affine_map<{} -> ({})".format((str(tuple(input_map.values())).replace("'","")), d_map) + ">]"
        else:
            mapping_String += "affine_map<{} -> {}".format((str(tuple(input_map.values())).replace("'","")), (str(tuple(d_map)).replace("'",""))) + ">]"

        return mapping_String
                    


class ele_wise_fill_Builder:
    indentation_size = 4
    ele_wise_fill_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '{{lbtensor_op_var}} = "ta.labeled_tensor"{{dims_tensor_tuple}} : ({{tensor_type}}, {{ranges_tuple}}) -> {{tensor_type}}'
        + "\n"
        +'{{const_op_var}} = "ta.constant"() {value = dense<{{assigned_val}}> : {{tensor_type}}} : () -> {{tensor_type}}'
        + "\n"
        + '"ta.set_op"({{const_op_var}}, {{lbtensor_op_var}}) {__beta__ = 0.000000e+00 : f64} : ({{tensor_type}},{{tensor_type}}) -> ()'
        + "\n",
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self,tensor_to_label, lbtensor_op_var, const_op_var, assigned_val:numpy.ndarray, tensor_type, dims:list) -> None:
        self.tensor_to_label = tensor_to_label
        self.lbtensor_op_var = lbtensor_op_var
        self.const_op_var = const_op_var
        self.assigned_val = assigned_val
        self.tensor_type = tensor_type
        self.dims = dims
        pass


    def build(self):

        dims_tensor_tuple = "({},".format(self.tensor_to_label)
        ranges_tuple  = ""
        for i in range(len(self.dims)-1):
            dims_tensor_tuple += self.dims[i] + ","
            ranges_tuple += "!ta.range,"

        dims_tensor_tuple += self.dims[-1] +  ")"
        ranges_tuple += "!ta.range"

        return self.ele_wise_fill_wrapper_text.render(
            lbtensor_op_var = self.lbtensor_op_var,
            const_op_var = self.const_op_var,
            assigned_val = self.assigned_val.tolist(),
            tensor_type = self.tensor_type,
            dims_tensor_tuple = dims_tensor_tuple,
            ranges_tuple = ranges_tuple
        )
        
class Tensor_arithOp_builder:

    indentation_size = 4
   
    tensor_add_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.add"{{operators}}'
        +' : {{Tensor_types_tuple}}'
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    tensor_sub_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + ' = "ta.subtract"{{operators}}'
        +' : {{Tensor_types_tuple}}'
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self,tensor_operators:list,input_types:list,outputtype:str,op:str) -> None:
        self.tensor_operators = tensor_operators
        self.input_types = input_types
        self.outputtype = outputtype
        self.op = op
        pass

    def build(self):
        operators_tuple = str(tuple(self.tensor_operators)).replace("'", "")
        input_types_tuple = str(tuple(self.input_types)).replace("'", "")

        if(self.op == '+'):
            return self.tensor_add_wrapper_text.render(
                operators = operators_tuple,
                Tensor_types_tuple = input_types_tuple,
                outputtype = self.outputtype
            )

        elif(self.op == '-'):
            return self.tensor_sub_wrapper_text.render(
                operators = operators_tuple,
                Tensor_types_tuple = input_types_tuple,
                outputtype = self.outputtype
            )
        
        
class MLIRFunctionBuilder:
    _ops = {}

    default_indentation_size = 4
    indentation_delta_size = 2
    module_wrapper_text = jinja2.Template(
        "{{ aliases }}module {\n   {{ body }}\n}\n",
        undefined=jinja2.StrictUndefined,
    )
    function_wrapper_text = jinja2.Template(
        ("" * default_indentation_size)
        + "func.func {% if private_func %}private {% endif %}@{{func_name}}({{signature}}) -> {{return_type}} {"
        + "\n"
        + "{{statements}}"
        + "\n"
        + (" " * default_indentation_size)
        + "}",
        undefined=jinja2.StrictUndefined,
    )

    def __init__(
        self,
        func_name: str,
        #input_types: Sequence[Union[str, Type]],
        return_types: Sequence[Union[str, types_mlir.Type]],
        aliases: types_mlir.AliasMap = None,
    ) -> None:
        # TODO mlir functions can return zero or more results https://mlir.llvm.org/docs/LangRef/#operations
        # handle the cases where the number of return types is not 1
        if aliases is None:

            aliases = types_mlir.AliasMap()
        self.aliases = aliases

        # Build input vars and ensure all types are proper Types
        # inputs = []
        # for i, it in enumerate(input_types):
        #     it = Type.find(it, aliases)
        #     # Create initialized MLIRVar
        #     iv = MLIRVar(f"arg{i}", it)
        #     iv._initialized = True
        #     inputs.append(iv)
        return_types = [types_mlir.Type.find(rt, aliases) for rt in return_types]

        self.func_name = func_name
        #self.inputs = inputs
        self.return_types = return_types

        self.var_name_counter = itertools.count()
        self.function_body_statements: List[str] = []
        self.temporary_statement_lists: List[List[str]] = []

        # function_name -> (function_mlir_definition, input_mlir_types, return_mlir_type)
        self.needed_function_table: Dict[
            str, Tuple[str, List[str], str]
        ] = OrderedDict()

        self.indentation_level = 1
        self._initialize_ops()

    def _initialize_ops(self):
        for dialect, ops in self._ops.items():
            if dialect is None:
                attach_point = self
            else:
                attach_point = getattr(self, dialect, None)
                if attach_point is None:
                    attach_point = Dialect(dialect)
                    setattr(self, dialect, attach_point)

            for opclass in ops.values():

                def op(opclass, *args, **kwargs):
                    ret_val, mlir = opclass.call(self, *args, **kwargs)
                    self.add_statement(mlir)
                    return ret_val

                func = functools.partial(op, opclass)
                setattr(attach_point, opclass.name, func)
    
    #######################################
    # MLIR Generation/Compilation Methods #
    #######################################

    def get_mlir_module(self, make_private=False):
        """Get the MLIR text for this function wrapped in a MLIR module with
        declarations of external helper functions."""
        aliases = "\n".join(
            f"#{name} = {typ.to_pretty_string()}" for name, typ in self.aliases.items()
        )
        body = self.get_mlir(make_private=make_private)
        return self.module_wrapper_text.render(aliases=aliases, body=body)

    def get_mlir(self, make_private=True, include_func_defs=True) -> str:
        if include_func_defs:
            needed_function_definitions = "\n    ".join(
                func_def for func_def, _, _ in self.needed_function_table.values()
            )
        else:
            needed_function_definitions = ""

        if len(self.temporary_statement_lists) > 0:
            raise RuntimeError(
                "Cannot get MLIR code while using temporary statement storage."
            )
        joined_statements = "\n".join(self.function_body_statements)

        return_type = ", ".join(str(rt) for rt in self.return_types)
        if len(self.return_types) != 1:
            return_type = f"({return_type})"

        #signature = ", ".join(f"{var}: {var.type}" for var in self.inputs)

        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature="",
            return_type=return_type,
            statements=joined_statements,
        )


    def compile(self):
        # if engine is None:
        #     engine = self.engine

        # Force recompilation if name is already registered
        # if self.func_name in engine.name_to_callable:
        #     del engine.name_to_callable[self.func_name]

        mlir = self.get_mlir_module()

        # engine.add(mlir, passes)
        # func = engine[self.func_name]
        # func.builder = self

        return mlir

    
    ################################
    # MLIR Building Method Helpers #
    ################################

    @property
    def current_statement_list(self) -> List[str]:
        return (
            self.temporary_statement_lists[-1]
            if len(self.temporary_statement_lists) > 0
            else self.function_body_statements
        )
    
    def add_statement(self, statement: str) -> None:
        """In an ideal world, no human would ever call this method."""
        if not statement:
            return

        for line in map(str.strip, statement.split("\n")):
            self.current_statement_list.append(
                " " * self.default_indentation_size
                + " " * self.indentation_delta_size * self.indentation_level
                + line
            )
    

