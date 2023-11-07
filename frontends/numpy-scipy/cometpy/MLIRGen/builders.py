import jinja2
import functools
import itertools

from typing import Dict, List, Tuple, Sequence, Union
from collections import OrderedDict
from ast import operator
from cometpy.MLIRGen import types_mlir
from cometpy.MLIRGen.types import *


class Dialect:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name} dialect"


class MLIRFunctionBuilder:
    _ops = {}

    default_indentation_size = 4
    indentation_delta_size = 2
    module_wrapper_text = jinja2.Template(
        "{{ aliases }}module {\n   {{ body }}\nfunc.func private @quick_sort(memref<*xindex>, index)\n}\n",
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
        input_types,
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
        self.inputs = input_types
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
        signature = ", ".join(f"{ var[0].replace('%','%arg_')}: {var[1].replace('tensor', 'memref')}" if 'tensor' in var[1] else  f"{ var[0].replace('%', '%arg_') }: memref<1xf64>" if var[1] =="f64" else f"{ var[0]}: {var[1]}" for var in self.inputs)

        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature=signature,
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
    

class TensorSumBuilder:
    indentation_size = 4
    tensor_sum_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.reduce"{{operators}}'
        +' : ({{inputtype}})'
        +"-> {{output_types}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, lhs, operators, tensors_shapes, label_map):
        self.lhs = lhs
        self.tensors_shapes = tensors_shapes
        output_type = "tensor<{}xf64>".format("x".join(str(label_map[v][0]) for v in self.tensors_shapes[-1]))
        input_type = []
        self.input_type = "tensor<{}xf64>".format("x".join(str(label_map[v][0]) if label_map[v][1] == DENSE else '?' for v in self.tensors_shapes[0]))

        self.operators = self.operators = "({})".format(",".join("%t"+str(v) for v in operators))

    def build_op(self):
        output_type = "f64"
        # for t in self.tensors_shapes[:-1]:
        #     input_type.append("tensor<{}xf64>".format("x".join(str(v) for v in t)))
        # input_type = ",".join(input_type) 
        
        return self.tensor_sum_wrapper_text.render(
            lhs = self.lhs,
            operators = self.operators,
            inputtype = self.input_type,
            output_types = output_type
        )

class SetOp_Builder:
    indentation_size = 4
    beta_val = 0.0

    set_op_wrapper_text = jinja2.Template(
        ("" * indentation_size)
        + '"ta.set_op"(%temp_{{input}},%t{{dest}}) {__beta__ = {{beta}} : f64} : ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, in_tensor, target, tensors_shapes, label_map, beta) :
        self.target = target
        self.in_tensor = in_tensor
        self.tensors_shapes =[]
        for l in tensors_shapes:
            self.tensors_shapes.append([ label_map[lbl][0] if label_map[lbl][1] == DENSE else '?' for lbl in l ]  )
        self.beta = "{:e}".format(beta)


    def build_op(self):
        output_type = "tensor<{}xf64>".format("x".join(str(v) for v in self.tensors_shapes[-1]))
        # input_type = []
        # for t in self.tensors_shapes[:-1]:
        #     input_type.append("tensor<{}xf64>".format("x".join(str(v) for v in t)))

        return self.set_op_wrapper_text.render(
            dest = self.target,
            input = self.in_tensor,
            beta = self.beta,
            outputtype= output_type,

        )

class ArithOp_Builder:
    formats_str = ['Dense', 'CSR', 'COO', 'CSC']
    indentation_size = 4
    beta_val = 0.0



    tc_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%temp_{{dest}} = "ta.mul"({{operators}})'
        + '{MaskType = "{{mask_type}}", ' 
        + '__alpha__ = 1.000000e+00 : f64, '  
        +"__beta__ = {{beta}} : f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}}, '
        +'operand_segment_sizes = array<i32:1, 1, {{lhs_dims}}, {{num_masks}}>, ' #[TODO] operand_segment_sizes should not be static
        +'semiring = "{{semiring}}"} : ' 
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}) {__beta__ = {{beta}} : f64} : ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )



    tensor_add_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%temp_{{dest}} = "ta.add"({{operators}})'
        +' {'
        +' Masktype = "none",'
        +' formats = [{{formats}}],'
        +' indexing_maps = {{indexing_maps}},'
        +' semiring = "noop_plusxy"'
        +' }'
        +' : ({{inputtype}})'
        +"-> {{outputtype}}"
        + "\n", 
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tensor_sub_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%temp_{{dest}} = "ta.subtract"({{operators}})'
        +' {'
        +' Masktype = "none",'
        +' formats = [{{formats}}],'
        +' indexing_maps = {{indexing_maps}},'
        +' semiring = "noop_minus"'
        +' }'
        +' : ({{inputtype}})'
        +"-> {{outputtype}}"
        + "\n",
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    elewisemult_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%temp_{{dest}} = "ta.elews_mul"({{operators}})'
        +' {__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = {{beta}}: f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}}, semiring = "{{semiring}}"} : '
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tranpose_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%temp_{{dest}} = "ta.transpose"({{operators}})'
        + '{__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = 0.000000e+00 : f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}},semiring = "plusxy_times"} : '
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, dest, input_tensors:list, tc_indices, formats: list, tensors_shapes, opr_type, label_map, mask=None, mask_type="none", mask_lbls = None, semiring=None, beta=0):
                #   dimslbls_to_map:list, input_array_dims_lbls:list, 
                #             target_dims_lbls:list,tensor_types:list,tc_indices:list,opr_type:str,op:str, formats:list) -> None:
        
        self.dest = dest
        self.operators = "{}".format(",".join("%t"+str(v) for v in input_tensors)+","+",".join("%i"+str(vv) for v in tensors_shapes for vv in v))
        self.tc_indices = tc_indices
        # self.dimslbls_to_map = dimslbls_to_map
        # self.input_array_dims_lbls = input_array_dims_lbls
        # self.target_dims_lbls = target_dims_lbls
        self.tensors_shapes =[]
        for l in tensors_shapes:
            self.tensors_shapes.append([ label_map[lbl][0] if label_map[lbl][1] == DENSE else '?' for lbl in l ]  )
        # self.tensors_shapes = [label_map[lbl][0] for lbl in tensors_shapes]
        self.opr_type = opr_type
        # self.op = op
        self.formats = formats
        self.mask = mask
        self.mask_type = mask_type
        if mask_lbls != None:
            self.mask_shape = [ label_map[lbl][0] if label_map[lbl][1] == DENSE else '?' for lbl in mask_lbls ]
            self.operators+=",%t"+str(self.mask)
        else:
            self.mask_shape = None
        self.semiring = semiring
        self.beta = "{:e}".format(beta)


    def build_op(self):
        output_type = "tensor<{}xf64>".format("x".join(str(v) for v in self.tensors_shapes[-1]))
        input_type = []
        for t in self.tensors_shapes[:-1]:
            input_type.append("tensor<{}xf64>".format("x".join(str(v) for v in t)))
        for t in self.tensors_shapes:
            for v in t:
                input_type.append("!ta.range")
        input_type = ",".join(input_type) 
        if self.mask_shape != None:
            input_type += ",tensor<{}xf64>".format("x".join(str(v) for v in self.mask_shape))
        # beta_val = ArithOp_Builder.get_beta_val(self.op)
        
        ops = self.tc_indices.split(',')
        iMap = {}
        i = 0
        if len(ops) > 1:
            op1 = ops[0]
            op2, res = ops[1].split('->')
        else:
            op2 = []
            op1,res = ops[0].split('->')
        indexing_map = []
        temp = []
        for l in op1:
            # if l not in iMap:
            iMap[l] = i
            temp.append(i)
            i+=1
        indexing_map.append(temp)

        if len(ops) > 1:
            temp = []
            for l in op2:
                if l not in iMap:
                    iMap[l] = i
                    temp.append(i)
                    i+=1
                else:
                    temp.append(iMap[l])

            indexing_map.append(temp)
        temp = []
        for l in res:
            temp.append(iMap[l])
        indexing_map.append(temp)
        indexing_maps = []

        for imap in indexing_map:
            indexing_maps.append("affine_map<({})->({})>".format(",".join(["d"+str(l) for l in range(i)]) , ",".join(["d"+str(l) for l in imap])))
        
        indexing_maps = str(indexing_maps).replace("'","")

        
        # Tensor contraction
        if self.opr_type == 'c': 
            semiring = "plusxy_times"
            if self.semiring != None:
                s1, s2 = self.semiring.split(",")
                if s1 == "+":
                    semiring = "plusxy_"
                elif s1 == "any":
                    semiring = "any_"
                elif s1 == "min":
                    semiring = "minxy_"
                if s2 == "*":
                    semiring += "times"
                elif s2 == "pair":
                    semiring += "pairxy"
                elif s2 == "first":
                    semiring += "first"
                elif s2 == "+":
                    semiring += "plusxy"
                elif s2 == "second":
                    semiring += "second"


            return self.tc_decl_wrapper_text.render(
                    dest = self.dest,
                    operators = self.operators,
                    indexing_maps = indexing_maps,
                    inputtype = input_type,
                    outputtype = output_type,
                    # beta =  self.beta_val,
                    formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                    lhs_dims = sum([len(t) for t in self.tensors_shapes ]),
                    semiring = semiring,
                    mask=self.mask,
                    mask_type = self.mask_type,
                    num_masks = 0 if self.mask == None else 1,
                    beta = self.beta,
                )
        # Add
        elif(self.opr_type == '+'):
            return self.tensor_add_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                indexing_maps = indexing_maps
            )
        # Subtract
        elif(self.opr_type == '-'):
            return self.tensor_sub_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                indexing_maps = indexing_maps
            )
        # Elementwise mult
        elif(self.opr_type == '*'):
            semiring = "noop_times"
            if self.semiring != None:
                if self.semiring == "min":
                    semiring = "noop_minxy"
                elif self.semiring == "-":
                    semiring = "noop_minus"
                elif self.semiring == "+":
                    semiring = "noop_plusxy"
                elif self.semiring == "*":
                    semiring = "noop_times"
            return self.elewisemult_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                indexing_maps = indexing_maps,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                semiring = semiring,
                beta =  self.beta_val
            )
        # Transpose
        elif(self.opr_type == "t"):
            return self.tranpose_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                indexing_maps = indexing_maps,
                inputtype = input_type,
                outputtype = output_type,
                # beta =  self.beta_val,
                formats = '"{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
            )
    # def get_beta_val(op):
    #     if(op == '='):
    #         beta_val = '0.000000e+00'
    #     elif(op == '+='):
    #         beta_val = '1.000000e+00'
    #     elif(op == '-='):
    #         beta_val = '-1.000000e+00'
    #     elif(op == '+' or op == '-'):
    #         beta_val = '0.000000e+00'
    #     return beta_val
    

class Tensor_Decl_Builder:
    formats = ['Dense', 'CSR', 'COO', 'CSC']
    indentation_size = 4

    tensor_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.{{decl}}_tensor_decl"{{dims_tuple}}'
        + '{format = {{format}}} : '
        +"{{ranges_tuple}} -> "
        + "{{inputtype}}"
        + "\n" 
        +'"ta.fill{{where}}"(%t{{lhs}}) {{value}} : ({{inputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tensor_decl_wrapper_text_no_fill = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.{{decl}}_tensor_decl"{{dims_tuple}}'
        + '{format = {{format}}} : '
        +"{{ranges_tuple}} -> "
        + "{{inputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, lhs, decl_vars:list, input_shape: str, format, dtype, label_map, is_input)->None:
        self.lhs = lhs
        self.inputtype = "tensor<{}x{}>".format("x".join(str(label_map[v][0]) if label_map[v][1] == DENSE else '?' for v in input_shape), dtype)
        self.decl_vars = decl_vars
        self.format = format
        self.is_input = is_input


    def build_tensor(self):
        dims_tuple = "({})".format(",".join("%i"+str(v) for v in self.decl_vars))
        ranges_tuple = "({})".format(",".join(["!ta.range"]* len(self.decl_vars)))
        # ranges_tuple = "("
        # for i in range(len(self.decl_vars)-1):
        #     dims_tuple += self.decl_vars[i] + ","
        #     ranges_tuple += "!ta.range,"
        
        # dims_tuple += self.decl_vars[-1] +  ")"
        # ranges_tuple += "!ta.range)"
        if not self.format == DENSE:
            where = "_from_file"
            format = '"{}" , temporal_tensor = false'.format(self.formats[self.format])
            value = '{filename = "SPARSE_FILE_NAME0", readMode = 1 : i32}'
        else:
            where = ""
            format = '"{}"'.format(self.formats[self.format])
            value = '{value = 0.0 : f64}'

        if self.is_input or self.format == DENSE:
            return self.tensor_decl_wrapper_text.render(
                lhs = self.lhs,
                dims_tuple = dims_tuple,
                ranges_tuple = ranges_tuple,
                format = format,
                inputtype = self.inputtype,
                decl = "dense" if self.format == DENSE else "sparse",
                where = where,
                value = value
            )
        else:
            return self.tensor_decl_wrapper_text_no_fill.render(
                lhs = self.lhs,
                dims_tuple = dims_tuple,
                ranges_tuple = ranges_tuple,
                format = format,
                inputtype = self.inputtype,
                decl = "dense" if self.format == DENSE else "sparse",
            )

class PrintBuilder:
    indentation_size = 4

    tensor_print_text = jinja2.Template(
        ("" * indentation_size)
        +'"ta.print"(%t{{tensor}}) : ({{outtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, operand, input_labels, dtype, label_map):
        self.operand = operand[0]
        self.outtype = "x".join(str(label_map[v][0]) if label_map[v][1] == DENSE else '?' for v in input_labels[0])
        if len(self.outtype) > 0:
            self.outtype = "tensor<{}x{}>".format(self.outtype, dtype)
        else:
            self.outtype = dtype
    def build_op(self):
        return self.tensor_print_text.render(
            tensor = self.operand,
            outtype = self.outtype,
        )