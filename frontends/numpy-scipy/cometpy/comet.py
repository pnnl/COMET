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

# from _ast import  Attribute, BinOp, Call, Constant
import ast_comments as ast
import inspect
# import comet
import jinja2
import numpy as np
import scipy as sp
from cometpy.MLIRGen import lowering
from cometpy.MLIRGen import ops
from cometpy.MLIRGen import types
from cometpy.MLIRGen.types import *
#import time

def lower_einsum_expresion(args, kwargs, visitor):
    char_index_to_symbol_index = {}
    all_input_indices, res_indices = args[0].value.split('->')
    inputs_indices = all_input_indices.split(',')
    inputs_symbol_indices = []
    for input_indices in inputs_indices:
        input_symbol_indices = []
        for c in input_indices:
            if c not in char_index_to_symbol_index:
                char_index_to_symbol_index[c] = visitor.build(ops.TensorIndexLabelOp()).results[0]
            input_symbol_indices.append(char_index_to_symbol_index[c])
        inputs_symbol_indices.append(input_symbol_indices)
    
    operands = [visitor.visit(arg) for arg in args[1:len(inputs_indices)+1]]
    semiring = None
    if len(inputs_indices) < len(args[1:]):
        semiring = args[-1].value
    elif kwargs and 'semiring' == kwargs[0].arg:
        semiring = kwargs[0].value.value

    all_same = True
    for indices in inputs_indices[1:]:
        if indices != inputs_indices[0]:
            all_same = False
    print(res_indices, inputs_indices[0])
    if len(inputs_indices) == 1 : # Tranpose
        res_symbol_indices = []
        
        for res_indice in res_indices:
            res_symbol_indices.append(char_index_to_symbol_index[res_indice])
        input = operands[0]
        input_indices = inputs_symbol_indices[0]
        res_format = visitor.sp_mattr_conversions[input.type.format]
        res = visitor.build(ops.TensorTransposeOp(input, input_indices, res_symbol_indices, res_format)).results[0]
        lhs = res

    elif all_same and inputs_indices[0] == res_indices: # Elementwise multiplication
        if semiring:
            if semiring == "min":
                semiring = "noop_minxy"
            elif semiring == "-":
                semiring = "noop_minus"
            elif semiring == "+":
                semiring = "noop_plusxy"
            elif semiring == "*":
                semiring = "noop_times"
        
        lhs, lhs_indices = operands[0], inputs_symbol_indices[0]
        for rhs in operands[1:]:
            res_format = visitor.sp_elw_mult_conversions[lhs.type.format][rhs.type.format]
            if semiring:
                res = visitor.build(ops.TensorMulOp(lhs, rhs, lhs_indices, res_format, semiring=semiring)).results[0]
            else:
                res = visitor.build(ops.TensorMulOp(lhs, rhs, lhs_indices, res_format)).results[0]
            lhs = res

    else: # Tensor contraction
        if semiring:
            s1, s2 = semiring.split(",")
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

        lhs, lhs_indices = operands[0], inputs_symbol_indices[0]
        for (i, (rhs, rhs_indices)) in enumerate(zip(operands[1:], inputs_symbol_indices[1:])):
            if i == len(operands) - 2:
                res_symbol_indices = []
                for symbol in res_indices:
                    res_symbol_indices.append(char_index_to_symbol_index[symbol])
            else:
                res_symbol_indices = [] 
                for indice in lhs_indices:
                    if indice not in rhs_indices:
                        res_symbol_indices.append(indice)
                for indice in rhs_indices:
                    if indice not in lhs_indices:
                        res_symbol_indices.append(indice)

            res_format = visitor.sp_matmult_conversions[lhs.type.format][rhs.type.format]
            if semiring:
                print(res_format)
                res = visitor.build(ops.TensorMatMultOp(lhs, rhs, lhs_indices, rhs_indices, res_symbol_indices, res_format, semiring=semiring)).results[0]
            else:
                res = visitor.build(ops.TensorMatMultOp(lhs, rhs, lhs_indices, rhs_indices, res_symbol_indices, res_format)).results[0]
            lhs, lhs_indices = res, res_symbol_indices

    if lhs.type.format != DENSE:
        visitor.need_opt_comp_workspace = True
    return lhs
    # for c in res_indices:
    #     res_symbol_indices.append(visitor.build(TensorIndexLabelOp()))



class NewAstParser(ast.NodeVisitor):

    identation_default = 2
    def __init__(self,inputs):
        self.symbol_table = ops.SymbolTable()
        self.inputs = inputs
        self.insertion_point = [self]
        self.identation = 0
        self.return_type = []
        self.pragma = ''
        self.body = []
        self.need_opt_comp_workspace = False
        self.build(ops.ModuleOp())


        # Output formats when multiply matrices of different formats
        self.sp_matmult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }

        self.sp_indices_results = {
            'i64': {
                'i64': 'i64',
                'i32': 'i64',
                None: 'i64',
            },

            'i32': {
                'i64': 'i64',
                'i32': 'i32',
                None: 'i32',
            }
            ,
            None: {
                'i64': 'i64',
                'i32': 'i32',
                None: None,
            }
        }

        # Output formats when transposing matrices of different formats
        # self.sp_mattr_conversions = {
        #     CSR : CSC,
        #     COO : COO,
        #     DENSE : DENSE
        # }

        self.sp_mattr_conversions = {
            CSR : CSR,
            COO : COO,
            DENSE : DENSE
        }

        # Output formats when elwise mult matrices of different formats
        # self.sp_elw_mult_conversions = {
        #     CSR: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO,
        #     },
        #     COO: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO
        #     },
        #     DENSE: {
        #         CSR : DENSE,
        #         COO : DENSE,
        #         DENSE : DENSE,
        #     }
        # }
        self.sp_elw_mult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : COO
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }
        # Output formats when elwise add or subtract matrices of different formats
        # Add and subtract is almost the same as elwise mult with two differences
        self.sp_elw_add_sub_conversions = self.sp_elw_mult_conversions


        # self.sp_elw_add_sub_conversions[CSR][DENSE] = DENSE
        # self.sp_elw_add_sub_conversions[COO][DENSE] = DENSE

        self.sp_elw_add_sub_conversions[CSR][DENSE] = CSR
        self.sp_elw_add_sub_conversions[COO][DENSE] = COO


    def dump(self):
        return "\n".join([stmt.dump() for stmt in self.body])

    def build(self, op):
        # self.mlir_code.append(" " * self.identation + op.dump())
        self.insertion_point[-1].body.append(op)
        if op.startsBody:
            self.identation += NewAstParser.identation_default
            op.identation = self.identation
            self.insertion_point.append(op)
        elif op.endsBody:
            self.identation -= NewAstParser.identation_default
            # self.mlir_code.append(" " * self.identation  + "}")
            self.insertion_point.pop()
        return op

    def visit_FunctionDef(self, node):
        args = []
        for arg, val in zip(node.args.args, self.inputs):
            new_arg = ops.Symbol(ops.mlir_type_from_ndarray(val))
            self.symbol_table.insert(arg.arg, new_arg)
            args.append(new_arg)
        funcOp = self.build(ops.FuncOp(node.name, args, [], False))
        # for arg in args:
        #     self.build(ops.TensorPrintOp(arg))
        for stmt in node.body:
            self.visit(stmt)


        if self.return_type:
            funcOp.return_types = self.return_type
        else:
            self.build(ops.ReturnOp([]))
        # self.mlir_code.append("" *self.identation + "}")

    def visit_Comment(self, node):
        if node.value.startswith('#pragma'):
            self.pragma = node.value

    def visit_Assign(self, node):
        self.pragma = ''
        rhs = self.visit(node.value)
        if  isinstance(node.targets[0], ast.Name):
            self.symbol_table.insert(node.targets[0].id, rhs)
        elif isinstance(node.targets[0], ast.Subscript):
            mem = self.visit(node.targets[0].value)
            indices = self.visit(node.targets[0].slice)
            if indices: # 
                if not isinstance(indices, list):
                    indices = [indices]
                if isinstance(mem.type, TensorType):
                    mem = self.build(ops.ToMemrefOp(mem))
                    self.build(ops.StoreOp(rhs, mem.results[0], indices))
            else:
                self.build(ops.TensorSetOp(rhs, mem, 0.0))
                self.build(ops.TensorPrintOp(mem))
            # self.mlir_code.append(store.dump())
        # lhs =  self.visit(node.targets[0].)
        # if indices:
        #     store = StoreOp(load.results[0], lhs)
        

    def visit_BinOp(self, node):
        self.pragma = ''

        left = self.visit(node.left)
        right = self.visit(node.right)
        val = None
        res_format = DENSE
        if isinstance(node.op, ast.Add):
            if isinstance(left.type, ShapedType):
                indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in left.type.shape] 
                res_format = self.sp_elw_add_sub_conversions[left.type.format][right.type.format]
                val = self.build(ops.TensorAddOp(left, right, indices, res_format))
            else:
                val = self.build(ops.AddOp(left, right))
        elif isinstance(node.op, ast.Sub):
            if isinstance(left.type, ShapedType):
                indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in left.type.shape] 
                res_format = self.sp_elw_add_sub_conversions[left.type.format][right.type.format]
                val = self.build(ops.TensorSubOp(left, right, indices, res_format))
            else:
                val = self.build(ops.SubOp(left, right))
        elif isinstance(node.op, ast.Mult):
            if isinstance(left.type, ShapedType):
                indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in left.type.shape] 
                res_format = self.sp_elw_mult_conversions[left.type.format][right.type.format]
                val = self.build(ops.TensorMulOp(left, right, indices, res_format))
            else:
                val = self.build(ops.MulOp(left, right))
        elif isinstance(node.op, ast.MatMult):
            print(left.type)
            print(right.type)
            assert(isinstance(left.type, ShapedType) and isinstance(right.type, ShapedType))
            lhs_indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in left.type.shape] 
            rhs_indices = [lhs_indices[-1]] + [self.build(ops.TensorIndexLabelOp()).results[0] for d in right.type.shape[1:]]
            if len(lhs_indices) ==2 and len(rhs_indices)  == 2:
                res_indices = [lhs_indices[0], rhs_indices[1]]
            elif len(lhs_indices) == 2 and len(rhs_indices)  == 1:
                res_indices = [lhs_indices[0]]
            elif len(lhs_indices) == 1 and len(rhs_indices)  == 2:
                res_indices = [rhs_indices[1]]
            res_format = self.sp_matmult_conversions[left.type.format][right.type.format]
            val  = self.build(ops.TensorMatMultOp(left, right, lhs_indices, rhs_indices, res_indices, res_format))
        if res_format != DENSE:
            self.need_opt_comp_workspace = True
        if val:
            return val.results[0]



    def visit_For(self, node):
        pragma = self.pragma
        assert(isinstance(node.iter, ast.Call) and node.iter.func.id == 'range')
        if len(node.iter.args) == 1:
            lb = self.build(ops.ConstantOp(0))

            lb = lb.results[0]
            ub = self.visit(node.iter.args[0])
            step = self.build(ops.ConstantOp(1))
            step = step.results[0]
        elif len(node.iter.args) == 2:
            lb = self.visit(node.iter.args[0])
            ub = self.visit(node.iter.args[1])
            step = self.build(ops.ConstantOp(1))
            step = step.results[0]
        else:
            lb = self.visit(node.iter.args[0])
            ub = self.visit(node.iter.args[1])
            step = self.visit(node.iter.args[2])
        if pragma == "#pragma parallel":
            forOp = self.build(ops.ForAllOp(lb, ub, step))
        else:
            forOp = self.build(ops.ForOp(lb, ub, step))
        self.symbol_table.insert(node.target.id, forOp.iv)
        for stmt in node.body:
            self.visit(stmt)
        if pragma == "#pragma parallel":
            self.build(ops.ReduceOp([]))
        else:
            self.build(ops.YieldOp([]))
        


    def visit_Subscript(self, node):
        self.pragma = ''
        
        mem = self.visit(node.value)
        assert(mem != None)
        indices = self.visit(node.slice)
        if not isinstance(indices, list):
            indices = [indices]
        if isinstance(node.value.ctx, ast.Load):
            if isinstance(mem.type, TensorType):
                mem = self.build(ops.ToMemrefOp(mem)).results[0]
            load = self.build(ops.LoadOp(mem, indices))
            return load.results[0]
        # elif isinstance(node.value.ctx, ast.Store):
        #     store = StoreOp(mem, indices)
        #     self.mlir_code.append(store.dump())

    def visit_Tuple(self, node):
        elts = [self.visit(elt) for elt in node.elts]
        return elts



    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            value = self.visit(node.func.value)
            if isinstance(value, ops.Symbol):
                values = [value]
            else :
                assert(value == 'comet')
            attr = node.func.attr
        elif isinstance(node.func, ast.Name):
            values = []
            attr = self.visit(node.func.id)

        if attr == 'sum':
            res = self.build(ops.TensorSumOp(*values)).results[0]
            return res
        elif attr == 'transpose':
            indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in values[0].type.shape]
            transpose_indices = [indices[1], indices[0]]
            format = None
            if isinstance(values[0].type, TensorType) or isinstance(values[0].type, MemrefType):
                format = DENSE
            elif isinstance(values[0].type, types.TASparseTensorType):
                format = values[0].type.format
            res = self.build(ops.TensorTransposeOp(values[0], indices, transpose_indices,  format)).results[0]
            return res
        elif attr == 'einsum':
            res = lower_einsum_expresion(node.args, node.keywords, self)
            return res
        elif attr == 'multiply':
            indices = [self.build(ops.TensorIndexLabelOp()).results[0] for d in values[0].type.shape]
            for arg in node.args:
                values.append(self.visit(arg))    
            res_format = self.sp_elw_mult_conversions[values[0].type.format][values[1].type.format]
            res = self.build(ops.TensorMulOp(values[0], values[1], indices, res_format)).results[0]
            return res
            


    def visit_Name(self, node):
        self.pragma = ''


        symbol = self.symbol_table.find(node.id)
        if symbol:
            return symbol
        else:
            return node.id

    def visit_Constant(self, node):
        self.pragma = ''

        val = None
        if isinstance(node.value, int):
            val = self.build(ops.ConstantOp(node.value))
        elif isinstance(node.value, float):
            val = self.build(ops.ConstantOp(node.value))
        if val:
            return val.results[0]

    def visit_Return(self, node):
        ret = self.visit(node.value)
        # self.build(ops.TensorPrintOp(ret))
        if not isinstance(ret, list):
            ret = [ret]
        retOp = self.build(ops.ReturnOp(ret))
        self.return_type = [ret.type for ret in retOp.results]


class NewVisitor(ast.NodeVisitor):

    def __init__(self,inputs):
        self.no_assign = False
        self.tsemantics = {}
        self.isemantics = {}
        self.tsymbols = {"comet": None}
        self.tcurr = 0
        self.ops = []
        self.declarations = []
        self.in_args = []
        self.returns = []
        self.mask = None
        self.need_opt_comp_workspace = False
        self.dcurr = 0
        self.inputs = inputs
        self.constants = {}
        self.valueToIndex = {}
        self.currValToIndex = {}
        self.currIndexLabel = 0

        # Output formats when multiply matrices of different formats
        self.sp_matmult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }

        self.sp_indices_results = {
            'i64': {
                'i64': 'i64',
                'i32': 'i64',
                None: 'i64',
            },

            'i32': {
                'i64': 'i64',
                'i32': 'i32',
                None: 'i32',
            }
            ,
            None: {
                'i64': 'i64',
                'i32': 'i32',
                None: None,
            }
        }

        # Output formats when transposing matrices of different formats
        # self.sp_mattr_conversions = {
        #     CSR : CSC,
        #     COO : COO,
        #     DENSE : DENSE
        # }

        self.sp_mattr_conversions = {
            CSR : CSR,
            COO : COO,
            DENSE : DENSE
        }

        # Output formats when elwise mult matrices of different formats
        # self.sp_elw_mult_conversions = {
        #     CSR: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO,
        #     },
        #     COO: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO
        #     },
        #     DENSE: {
        #         CSR : DENSE,
        #         COO : DENSE,
        #         DENSE : DENSE,
        #     }
        # }
        self.sp_elw_mult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : COO
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }
        # Output formats when elwise add or subtract matrices of different formats
        # Add and subtract is almost the same as elwise mult with two differences
        self.sp_elw_add_sub_conversions = self.sp_elw_mult_conversions


        # self.sp_elw_add_sub_conversions[CSR][DENSE] = DENSE
        # self.sp_elw_add_sub_conversions[COO][DENSE] = DENSE

        self.sp_elw_add_sub_conversions[CSR][DENSE] = CSR
        self.sp_elw_add_sub_conversions[COO][DENSE] = COO

    def get_next_indexlabel_with_val(self, v) :
        if v not in self.valueToIndex:
            self.valueToIndex[v] = []
            self.valueToIndex[v].append(self.currIndexLabel)
            self.currValToIndex[v] = 1
            self.currIndexLabel += 1

        else:
            if self.currValToIndex[v] >= len(self.valueToIndex[v]) :
                self.valueToIndex[v].append(self.currIndexLabel)
                self.currIndexLabel += 1

            self.currValToIndex[v] += 1
        return self.valueToIndex[v][self.currValToIndex[v]- 1]

    def reset_indexlabel_with_val(self, v) :
        self.currValToIndex[v] -= 1

    def get_index_constant(self,v) :
        if v not in self.constants:
            self.constants[v] = self.dcurr
            self.declarations.append(
                {
                    "type": "C",
                    "value": v,
                    "id": self.dcurr,
                })
            self.dcurr += 1

        return self.constants[v]

    def visit_FunctionDef(self, node):
        for i, arg in enumerate(node.args.args):
            self.tsymbols[arg.arg] = self.tcurr
            format = ops.get_format(self.inputs[i])
            for s in self.inputs[i].shape :
                indices_type = None
                if format == DENSE :
                    indices_type = None
                elif format == COO:
                    indices_type = ops.dtype_to_mlir_type(self.inputs[i].coords[0].dtype)
                elif format == CSR:
                    indices_type = ops.dtype_to_mlir_type(self.inputs[i].indices.dtype)
                else:
                    raise Exception("Unexpected sparse matrix type")

                self.get_next_indexlabel_with_val(s)
                self.reset_indexlabel_with_val(s)

            self.tsemantics[self.tcurr] = {
                'indices_type': indices_type,
                'value_type' : ops.dtype_to_mlir_type(self.inputs[i].dtype),
                'shape': list(self.inputs[i].shape),
                'in_device': hasattr(self.inputs[i], '__cuda_array_interface__'),
                'format': format,
                'dimsSSA': [self.get_index_constant(d) for d in self.inputs[i].shape],
                'scalar': False,
                }
            self.in_args.append(self.tcurr)
            self.tcurr += 1
        for stmt in node.body:
            NewVisitor.visit(self, stmt)

    
    def visit_Assign(self, node): # We do not support multiple targets currently

        if isinstance(node.targets[0], ast.Subscript):
            id = node.targets[0].value.id
        else:
            id = node.targets[0].id


        if  isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].ctx, ast.Store) and not isinstance(node.targets[0].slice, ast.Tuple): 
            
            v = NewVisitor.visit(self, node.value)
            # self.tsymbols[id] = v
            if not isinstance(node.targets[0].slice, ast.Tuple):
                self.ops.append(
                {
                    "indices_type": [self.tsemantics[v]['indices_type']]*2,
                    "value_type": self.tsemantics[v]['value_type'],
                    "op_type": "=",
                    "shapes": [self.tsemantics[v]['shape']]*2,
                    "lhs": self.tsymbols[id],
                    "rhs": v,
                    "beta": True,
                })
                if self.tsemantics[self.tsymbols[id]]['format'] == DENSE and self.tsemantics[self.tsymbols[id]]['shape'] != 1:
                    self.ops.append(
                        {
                            "indices_type": [None],
                            "value_type": self.tsemantics[self.tsymbols[id]]['value_type'],
                            "op_type": "p",
                            "shapes": [self.tsemantics[self.tsymbols[id]]["shape"]],
                            "value_type": self.tsemantics[self.tsymbols[id]]["value_type"],
                            "operands": [self.tsymbols[id]]
                        })

        else:
            
            if isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].ctx, ast.Store) and (isinstance(node.targets[0].slice, ast.Tuple)) :
                mask = node.targets[0].slice
                self.mask = mask
                print(slice)

            v = NewVisitor.visit(self, node.value)
            if self.tsemantics[v]['shape'] != 1:
                self.declarations.append({
                    'indices_type': self.tsemantics[v]['indices_type'], 
                    "value_type": self.tsemantics[v]['value_type'],
                    "type": "T",
                    "is_input": False,
                    "todo": "l",
                    "format": self.tsemantics[v]['format'],
                    "shape": self.tsemantics[v]['shape'],
                    "dimsSSA": [self.get_index_constant(d) for d in self.tsemantics[v]['shape']],
                    "id": self.tcurr,
                    })
                self.tsemantics[self.tcurr] = self.tsemantics[v]
                self.ops.append(
                {
                    "indices_type": [self.tsemantics[v]['indices_type']]*2,
                    "value_type": self.tsemantics[v]['value_type'],
                    "op_type": "=",
                    "shapes": [self.tsemantics[v]['shape']]*2,
                    "lhs": self.tcurr,
                    "rhs": v,
                    "beta": True,
                })
                self.tsymbols[id] = self.tcurr
                self.tcurr += 1
            else:
                self.tsymbols[id] = v


        

    def visit_AugAssign(self, node):
        id = node.target.id
        beta = 0
        no_assign = 0

        if isinstance(node.op, ast.Add):
            beta = 1
        elif isinstance(node.op, ast.Sub):
            beta = -1

        # if id in self.tsymbols:
        self.no_assign = beta
        no_assign = self.no_assign

        # Because comet handles +=,-= in a different way (using beta) for some operations
        # we need to specifically check for these cases.
        if isinstance(node.value, ast.Call) and node.value.func.attr == "einsum" :

            v = self.visit_Einsum_Call(node.value)
        
            if no_assign:
                self.ops.append(
                    {
                        "indices_type": [self.tsemantics[v]['indices_type'], self.tsemantics[self.tsymbols[id]]['indices_type']],
                        "value_type": self.tsemantics[v]['value_type'],
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.Call) and node.value.func.attr == "transpose" :
            v = self.visit_Call(node.value)
        
            if no_assign:
                self.ops.append(
                    {
                        "indices_type": [self.tsemantics[v]['indices_type'], self.tsemantics[self.tsymbols[id]]['indices_type']],
                        "value_type": self.tsemantics[v]['value_type'],
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            v = self.visit_BinOp(node.value)
            if no_assign:
                self.ops.append(
                    {
                        "indices_type": [self.tsemantics[v]['indices_type'], self.tsemantics[self.tsymbols[id]]['indices_type']],
                        "value_type": self.tsemantics[v]['value_type'],
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        # Not a special case. Just add the result to the LHS
        else:
            v = NewVisitor.visit(self, node.value)
            res = self.create_binOp(node, [self.tsymbols[id], v], no_assign)
            self.tsymbols[id] = res

    def visit_Call(self, node: ast.Call) :
        obj = None
        obj = NewVisitor.visit(self,node.func)
        if obj is not None:
            return self.visit_Method_Call(node, obj) 
        else:
            if node.func.attr == "einsum": # Handle comet.einsum
                return self.visit_Einsum_Call(node)


    def visit_Attribute(self, node: ast.Attribute) :
        return NewVisitor.visit(self, node.value)

    def visit_Name(self, node: ast.Name):
        return self.tsymbols[node.id]

    def visit_Constant(self, node: ast.Constant) :
        out_id = self.tcurr
        self.tsemantics[self.tcurr] = {'shape': [1,], 'format': DENSE, 'scalar': True}
        self.declarations.append(
            {
                "type": "V",
                "value": f"{node.value:e}",
                "todo": "l",
                "id": out_id,
            })
        self.tcurr +=1
        return out_id
    
    def create_binOp(self, node, operands, no_assign):
        op0_sems = self.tsemantics[operands[0]]
        op1_sems = self.tsemantics[operands[1]]
        if op0_sems['scalar'] and  op1_sems['scalar']:
            op = ''
            if isinstance(node.op, ast.Add):
                op = '+'
            elif isinstance(node.op, ast.Sub):
                op = '-'
            elif isinstance(node.op, ast.Mult):
                op = '*'
            elif isinstance(node.op, ast.Div):
                op = '/'
            else:
                raise "Unexpected operator {}".format(node.op)
            self.ops.append(
                {
                    "indices_type": [op1_sems['indices_type'], op0_sems['indices_type'], None],
                    "value_type": op0_sems['value_type'],
                    "op_type": "scalar",
                    "op": op,
                    "operands": operands[::-1],
                    "shapes": [op1_sems['shape'], op0_sems['shape'], [1,]],
                    "out_id": self.tcurr,
                }
            )
            self.tsemantics[self.tcurr] = {"value_type": op_semantics['value_type'], 'indices_type': None, 'shape': [1,], 'format': DENSE, 'scalar': True}

            self.tcurr += 1
            self.declarations.append({
                'indices_type': None, 
                "value_type": op_semantics['value_type'],
                "type": "V",
                "value": f"{0:e}",
                "is_input": False,
                "todo": "l",
                "format": DENSE,
                "shape": [1,],
                # "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                "id": self.tcurr,
                })
            self.ops.append(
                {
                    'indices_type': [None, None], 
                    "value_type": op_semantics['value_type'],
                    "op_type": "=",
                    "shapes": [[1,]]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]

            self.tcurr +=1
            return self.tcurr-1

        format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
        indices_type = self.sp_indices_results[op0_sems['indices_type']][op1_sems['indices_type']]
        if self.tsemantics[operands[0]]['format'] != DENSE:
            op_semantics = self.tsemantics[operands[0]]
        else:
            op_semantics = self.tsemantics[operands[1]]

        self.need_opt_comp_workspace = op0_sems['format'] == CSR and op1_sems['format'] == CSR
        if isinstance(node.op, ast.Add):
            self.ops.append(
            {
                'indices_type': [op0_sems['indices_type'], op1_sems['indices_type'], indices_type],
                "value_type": op_semantics['value_type'],
                "op_type": "+",
                "shapes": [op_semantics['shape']] * 3,
                "operands": operands,
                "op_ilabels": [[self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]] * 3,
                "beta": 0,
                "out_id": self.tcurr,
            })

            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)

            self.tsemantics[self.tcurr] = {'value_type': op_semantics['value_type'], 'indices_type': indices_type, 'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}
        elif isinstance(node.op, ast.Sub):
            self.ops.append(
                {
                    'indices_type': [op0_sems['indices_type'], op1_sems['indices_type'], indices_type],
                    "value_type": op_semantics['value_type'],
                    "op_type": "-",
                    "shapes": [op_semantics['shape']]* 3,
                    "operands": operands,
                    "op_ilabels": [[self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]] * 3,
                    "out_id": self.tcurr,
                    "beta": 0,
                })
            
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            self.tsemantics[self.tcurr] = {"value_type": op_semantics['value_type'], 'indices_type': indices_type, 'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}

        elif isinstance(node.op, ast.Mult):
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    'indices_type': [op0_sems['indices_type'], op1_sems['indices_type'], indices_type],
                    "value_type": op_semantics['value_type'],
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "op_ilabels": [in_ilabels] * 3,
                    "operands": operands,
                    "out_id": self.tcurr,
                    "semiring": None,
                    "beta": no_assign
                })
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            self.tsemantics[self.tcurr] = {'value_type': op_semantics['value_type'], 'indices_type': indices_type,'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}

        elif isinstance(node.op, ast.MatMult):
    # def visit_Bin_Einsum_Call(self, operands, llabels, mask,semiring, no_assing):

            mask = (None,None)
            if  self.mask != None:
                mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)
                self.mask = None
            op1 = self.tsemantics[operands[0]]
            op2 = self.tsemantics[operands[1]]
            if len(op1['shape']) == 2 and len(op2['shape']) == 2:
                indices = 'ij,jk->ik'
            elif len(op1['shape']) == 2 and len(op2['shape']) == 1:
                indices = 'ij,j->i'
            elif len(op1['shape']) == 1 and len(op2['shape']) == 2:
                indices = 'j,jk->k'
            
            return self.visit_Bin_Einsum_Call(operands, indices, mask, None, 0, False, {})

        self.tcurr +=1
        return self.tcurr-1

    def visit_BinOp(self, node: ast.BinOp) :
        no_assign = self.no_assign
        self.no_assign = False
        operands = [NewVisitor.visit(self, node.left), NewVisitor.visit(self, node.right)]
        return self.create_binOp(node, operands, no_assign)

    def visit_Method_Call(self, node: ast.Call, obj):
        no_assign = self.no_assign
        self.no_assign = False
        # operands = []
        out_id = self.tcurr
        op_semantics = self.tsemantics[obj]
        format = op_semantics['format']

        if node.func.attr == "transpose":
            out_format = self.sp_mattr_conversions[op_semantics['format']]
            self.tsemantics[self.tcurr] = {"value_type": op_semantics['value_type'], 'indices_type': op_semantics['indices_type'],  'shape': op_semantics['shape'][::-1], 'format': out_format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape'][::-1]], 'scalar': False}
            
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)

            out_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape'][::-1]]
            for d in op_semantics['shape'][::-1]:
                self.reset_indexlabel_with_val(d)

            self.ops.append(
                {
                    "indices_type": [op_semantics['indices_type']] * 2,
                    "value_type": op_semantics['value_type'],
                    "op_type": "t",
                    "shapes": [op_semantics['shape'], op_semantics['shape'][::-1]],
                    "operands": [obj],
                    "op_ilabels": [in_ilabels, out_ilabels],
                    "out_id": self.tcurr,
                    "beta": 0,
                })
            
        elif node.func.attr == "sum":
            self.tsemantics[self.tcurr] = {"value_type": op_semantics['value_type'], 'indices_type': None, 'shape': 1, 'format': DENSE, 'scalar': True}
            self.ops.append(
                {
                    "value_type": op_semantics['value_type'],
                    'indices_type': [op_semantics["indices_type"], None],
                    "op_type": "s",
                    "shapes": [op_semantics["shape"], [1,]],
                    "operands": [obj],
                    "out_id": self.tcurr,
                })
            
        elif node.func.attr == "multiply":
            op1 = NewVisitor.visit(self, node.args[0])
            op1_sems = self.tsemantics[op1]
            op_semantics = self.tsemantics[obj]

            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            indices_type = self.sp_indices_results[op_semantics['indices_type']][op1_sems['indices_type']]
            self.ops.append(
                {
                    "value_type": op_semantics['value_type'],
                    "indices_type": [op_semantics['indices_type'], op1_sems['indices_type'], indices_type],
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "operands": [obj, op1],
                    "op_ilabels": [in_ilabels] * 3,
                    "out_id": self.tcurr,
                    "semiring": None,
                    "beta": no_assign,
                })

            format = self.sp_elw_mult_conversions[op_semantics['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {"value_type": op_semantics['value_type'], "indices_type": indices_type, 'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape'][::-1]], 'scalar': False}

        self.tcurr +=1

        return self.tcurr-1

    def visit_Return(self, node):
        obj = NewVisitor.visit(self, node.value)
        self.returns.append(obj)
        if self.tsemantics[obj]['format'] == DENSE and self.tsemantics[obj]['shape'] != 1:
            self.in_args.append(obj)
            self.ops.append(
                {
                    "indices_type": [None],
                    "value_type": self.tsemantics[obj]['value_type'],
                    "op_type": "p",
                    "shapes": [self.tsemantics[obj]["shape"]],
                    "value_type": self.tsemantics[obj]["value_type"],
                    "operands": [obj]
                })
        if self.tsemantics[obj]['format'] != DENSE or self.tsemantics[obj]['shape'] == 1:
            self.ops.append(
            
                {
                    "indices_type": [self.tsemantics[obj]['indices_type']],
                    "value_type": self.tsemantics[obj]['value_type'],
                    "op_type": "r",
                    "shapes": [self.tsemantics[obj]["shape"]],
                    "value_type": self.tsemantics[obj]["value_type"],
                    "operands": [obj]
                })

    def visit_Bin_Einsum_Call(self, operands, llabels, mask,semiring, beta, no_assign, lbls_to_ilbls):

        ops = llabels.split('->')[0].split(',')
        res = list(llabels.split('->')[1])
        # ret_num = []
        all_real_lbls = []
        all_dims = self.tsemantics[operands[0]]['shape'][:] # copy
        all_lbls = list(ops[0])
        if len(ops) > 1:
            for i,l in enumerate(list(ops[1])):
                if l not in all_lbls:
                    all_lbls.append(l)
                    all_dims.append(self.tsemantics[operands[1]]['shape'][i])
        
        lbls_to_dims = {}
        ilbls_seen = set()

        for d,l in zip(all_dims, all_lbls):
            lbls_to_dims[l] = d
            if l not in lbls_to_ilbls:
                lbls_to_ilbls[l] = self.get_next_indexlabel_with_val(d)
        
        for d,l in zip(all_dims, all_lbls):
            self.reset_indexlabel_with_val(d)


        ret_num = [i for x in res for  i,v in enumerate(all_lbls) if x == v]
        in0_ilabels = [lbls_to_ilbls[l] for l in list(ops[0])]
        in1_ilabels = []
        if len(ops) > 1:
            in1_ilabels = [lbls_to_ilbls[l] for l in list(ops[1])]
        
        format = self.tsemantics[operands[0]]['format']
        indices_type = self.tsemantics[operands[0]]['indices_type']
        if format != DENSE:
                self.need_opt_comp_workspace = True

        all_dims = self.tsemantics[operands[0]]['shape'][:]
        all_lbls = in0_ilabels[:]
        if len(ops) > 1:
            for l,d in zip(in1_ilabels,self.tsemantics[operands[1]]['shape']):
                if l not in all_lbls:
                    all_lbls.append(l)
                    all_dims.append(d)


        shape = [all_dims[v] for v in ret_num]
        labels = [all_lbls[v] for v in ret_num]
        if len(ops) > 1:
            for op in operands[1:]:
                format = self.sp_matmult_conversions[format][self.tsemantics[op]['format']]
                indices_type = self.sp_indices_results[indices_type][self.tsemantics[op]['indices_type']]

                if format != DENSE:
                    self.need_opt_comp_workspace = True

            self.ops.append(
                {
                    "indices_type": [self.tsemantics[operands[0]]['indices_type'], self.tsemantics[operands[1]]['indices_type'],indices_type], 
                    "value_type": self.tsemantics[operands[0]]['value_type'],
                    "op_type": "c",
                    "operands": operands,
                    "op_ilabels": [in0_ilabels, in1_ilabels, labels],
                    "shapes": [self.tsemantics[operands[0]]['shape'], self.tsemantics[operands[1]]['shape'], shape],
                    "out_id": self.tcurr,
                    "mask": mask,
                    "semiring": semiring,
                    "beta": beta
                })
                
        elif len(ops) == 1:
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in self.tsemantics[operands[0]]['shape']]
            for d in self.tsemantics[operands[0]]['shape']:
                self.reset_indexlabel_with_val(d)
            
            out_ilabels = [self.get_next_indexlabel_with_val(d) for d in shape]
            for d in shape:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "value_type": self.tsemantics[operands[0]]['value_type'], 
                    "op_type": "t",
                    "indices_type": [self.tsemantics[operands[0]]['indices_type'], indices_type],
                    "shapes": [self.tsemantics[operands[0]]['shape'], shape],
                    "operands": operands,
                    "out_id": self.tcurr,
                    "beta": beta,
                    "op_ilabels": [in_ilabels, out_ilabels]
                })
                
        self.tsemantics[self.tcurr] = {'value_type': self.tsemantics[operands[0]]['value_type'],"indices_type": indices_type, 'shape': shape, 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in shape], 'scalar': False}
        self.tcurr += 1

        return  self.tcurr - 1

    def visit_Einsum_Call(self, node: ast.Call):
        no_assign =  self.no_assign
        self.no_assign = False
        out_id = self.tcurr
        mask = (None, None)
        # mask_type = "none"
        semiring = None
        if  self.mask != None:
            mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)

            self.mask = None
        iLabels = node.args[0].value
        ops, res = iLabels.split('->')
        ops = ops.split(',')
        res = list(res)
        if len(node.keywords) > 0:
            for key in node.keywords:
                if key.arg == 'mask':
                    mask = NewVisitor.visit(self, key.value)

                    mask = (mask, "pull")
                elif key.arg == 'semiring':
                    semiring = key.value.value
                elif key.arg == 'mask_type':
                    mask = (mask[0], key.value.value)

        operands = []
        if len(node.args) > 1:
            for arg in node.args[1:]:
                operands.append(NewVisitor.visit(self, arg))
        if semiring != None and len(semiring.split(',')) == 1:
            op0_sems = self.tsemantics[operands[0]]
            op1_sems = self.tsemantics[operands[1]]
            

            format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
            indices_type = self.sp_indices_results[op0_sems['indices_type']][op1_sems['indices_type']]
            if self.tsemantics[operands[0]]['format'] != DENSE:
                op_semantics = self.tsemantics[operands[0]]
            else:
                op_semantics = self.tsemantics[operands[1]]

            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "indices_type": [op0_sems['indices_type'], op1_sems['indices_type'], indices_type],
                    "value_type": op_semantics['value_type'],
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "operands": operands,
                    "op_ilabels": [in_ilabels] * 3,
                    "out_id": self.tcurr,
                    "semiring": semiring,
                    "beta": no_assign
                })
                
                # ("*", operands, indices+','+indices+'->'+indices, self.tcurr, semiring))
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            indices_type = self.sp_indices_results[op0_sems['indices_type']][op1_sems['indices_type']]
            self.tsemantics[self.tcurr] = {'value_type' : op_semantics['value_type'], 'indices_type': indices_type, 'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False }
            self.tcurr += 1

            return self.tcurr-1
        if len(operands) > 1: # Contraction
            loperand = operands[0]
            lops = ops[0]
            saved_no_assign = no_assign
            no_assign = True
            lbls_to_ilbls = {}
            for i in range(1, len(operands)):
                all_lbls = list(lops)
                for l in list(ops[i]):
                    if l not in all_lbls:
                        all_lbls.append(l)

                rh1_lbls_num = [curr for x in list(lops) for  curr,v in enumerate(all_lbls) if x == v]
                rh2_lbls_num = [curr for x in list(ops[i]) for  curr,v in enumerate(all_lbls) if x == v]
                all_lbls_num = rh1_lbls_num[:] # Copy
                for n in rh2_lbls_num:
                    if n not in all_lbls_num:
                        all_lbls_num.append(n)

                sum = set(rh1_lbls_num) & set(rh2_lbls_num) 
                all = set(rh1_lbls_num) | set(rh2_lbls_num) 
                ret = all - sum

                ret_num = [i for x in ret for  i,v in enumerate(all_lbls_num) if x == v]
                ret = [all_lbls[x] for x in ret_num]
                    



                if i == len(operands) -1 :
                    ret = res
                    no_assign = saved_no_assign
                tid = self.visit_Bin_Einsum_Call([loperand,operands[i]], lops+","+ops[i]+"->"+"".join(ret), mask, semiring, saved_no_assign, no_assign, lbls_to_ilbls)
                loperand = tid
                lops = "".join(ret)
            return tid
        elif len(operands) == 1: # transpose

            loperand = operands[0]
            lops = ops[0]
            ret = res
            tid = self.visit_Bin_Einsum_Call([loperand], lops+"->"+"".join(ret), mask, semiring, no_assign, no_assign, {})
            lops = "".join(ret)
            return tid


#Wrapper function. The input function (in the form of an object) is passed an arguement to this function.
def compile(flags, target:str = "cpu", with_jit=True):
    def innerfunc(func):

        def wrapper(*pos_args, **kwargs):
            func_str = ast.parse(inspect.getsource(func))
            parsed_func = ast.parse(func_str)
            func_def = parsed_func.body[0]
            new_v = NewAstParser([*pos_args])
            new_v.visit(parsed_func)
            # for line in new_v.mlir_code:
            #     print(line)
            # exit(0)
            '''
            v = NewVisitor([*pos_args])
            v.visit(parsed_func)
            in_types = []
            for arg in v.in_args:
                if isinstance(v.tsemantics[arg]['shape'], int):
                    in_types.append(("%t"+str(arg), "tensor<1x{}>".format(v.tsemantics[arg]['value_type'])))
                else:
                    if v.tsemantics[arg]['format'] == DENSE:
                        if 'in_device' in v.tsemantics[arg] and v.tsemantics[arg]['in_device']:
                            in_types.append(("%t"+str(arg), "tensor<{}x{}> {{gpu.in_device}}".format("x".join(str(d) for d in v.tsemantics[arg]['shape']), v.tsemantics[arg]['value_type'])))
                        else:
                            in_types.append(("%t"+str(arg), "tensor<{}x{}>".format("x".join(str(d) for d in v.tsemantics[arg]['shape']), v.tsemantics[arg]['value_type'])))
                    else:
                        ssa = "!ta.sparse_tensor<{}, {}, {}".format(v.tsemantics[arg]['value_type'],v.tsemantics[arg]['indices_type'],"x".join(str(d) for d in v.tsemantics[arg]['shape']))
                        if v.tsemantics[arg]['format'] == CSR:
                            if v.tsemantics[arg]['in_device']:
                                ssa = ssa + ", d, unk, cu, unk> {{gpu.indevice}}"
                            else:
                                ssa = ssa + ", d, unk, cu, unk>"
                            in_types.append(("%t"+str(arg), ssa))
                        elif v.tsemantics[arg]['format'] == COO:
                            if v.tsemantics[arg]['in_device']:
                                ssa = ssa + ", cn, unk, s, unk> {{gpu.indevice}}"
                            else:
                                ssa = ssa + ", cn, unk, s, unk>"
                            in_types.append(("%t"+str(arg), ssa))

            return_types=[]
            outputs = []
            ret_format = None
            ret_shape = None

            for r in  v.returns:
                ret = v.tsemantics[r]
                format = ret['format']
                shape = ret['shape']
                ret_shape = shape
                ret_format = format
                if format != DENSE:
                    type = "!ta.sparse_tensor<{}, {}, {}".format(ret['value_type'], ret['indices_type'], "x".join(str(d) for d in ret['shape']))
                    if format == CSR:
                        type += ", d, unk, cu, unk>"
                    elif format == COO:
                        type += ", cn, unk, s, unk>"
                    return_types.append(type)
                elif shape == 1:
                    type = ret['value_type']
                    return_types.append(type)

                if format == DENSE:
                    outputs.append(np.zeros(ret['shape'],dtype=mlir_type_to_dtype(ret['value_type'] )))
                elif format == CSR:
                    outputs.append(sp.sparse.csr_array(np.empty(ret['shape'], dtype=mlir_type_to_dtype(ret['value_type']))))
                elif format == COO:
                    outputs.append(sp.sparse.coo_array(np.empty(ret['shape'], dtype=mlir_type_to_dtype(ret['value_type']))))
                elif format == CSC:
                    outputs.append(sp.sparse.csc_array(np.empty(ret['shape'], dtype=mlir_type_to_dtype(ret['value_type']))))

            if flags:
                func_name = func_def.name + flags.replace('-','_').replace(' ','').replace('=','_')
            else:
                func_name = func_def.name
            irb = builders.MLIRFunctionBuilder(
                func_name,
                input_types=in_types,
                return_types=return_types,
            ) 

            for i in range(v.currIndexLabel):
                irb.add_statement('%i{} = "ta.index_label"() : () -> !ta.index'.format(i))


            dense_tensors = []
            scalars = []
            for dec in v.declarations:
                
                if dec["type"] == "T":
                    if dec["id"] in v.in_args:
                        continue
                    t = builders.Tensor_Decl_Builder(dec)
                    if dec["format"] == DENSE:
                        dense_tensors.append(t)
                    else:
                        irb.add_statement(t.build_tensor())
                elif dec["type"] == "C":
                    irb.add_statement('%d{} = arith.constant {} : index '.format(dec["id"], dec["value"]))
                elif dec["type"] == "V":
                    scalars.append('%t{} = ta.constant dense<{}> : tensor<1x{}> '.format(dec["id"], dec["value"], dec["value_type"]))


            for t in dense_tensors:
                if t not in v.in_args:
                    irb.add_statement(t.build_tensor())

            for t in scalars:
                irb.add_statement(t)

            for op in v.ops:
                if op["op_type"] == 'c':
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    if op["mask"][0] is not None :
                        op["mask"] = (op["mask"][0], op["mask"][1], v.tsemantics[op["mask"][0]]['shape'])
                        irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
                    else:
                        op["mask"] = (op["mask"][0], op["mask"][1], None)
                        irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
                elif op["op_type"] == 'scalar':
                    irb.add_statement(builders.ScalarOp_Builder(op).build_op())
                elif op["op_type"] == 's':
                    op["formats"] = [v.tsemantics[op["operands"][0]]['format']] 
                    irb.add_statement(builders.TensorSumBuilder(op).build_op())
                elif op["op_type"] == 'p':
                    op["formats"] = [v.tsemantics[op["operands"][0]]['format']] 
                    irb.add_statement(builders.PrintBuilder(op).build_op())
                elif op["op_type"] == 'r':
                    op["formats"] = [v.tsemantics[op["operands"][0]]['format']] 
                    irb.add_statement(builders.ReturnBuilder(op).build_op())
                elif op["op_type"] == '*':
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    irb.add_statement(builders.ArithOp_Builder(op).build_op())
                elif op["op_type"] == '=':
                    op["formats"] = [v.tsemantics[op['lhs']]['format']] + [v.tsemantics[op['rhs']]['format']] + [v.tsemantics[op['rhs']]['format']]
                    irb.add_statement(builders.SetOp_Builder(op).build_op())
                else:
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
            if ret_format is None  or (ret_format == DENSE and ret_shape != 1):
                irb.add_statement("return")

            '''
            arg_vals = new_v.inputs
            new_flags = flags
            if new_v.need_opt_comp_workspace:
                if new_flags:
                    new_flags = new_flags + ' --opt-comp-workspace'
                else:
                    new_flags = ' --opt-comp-workspace'
            if new_flags:
                func_name = func_def.name + new_flags.replace('-','_').replace(' ','').replace('=','_')
            else:
                func_name = func_def.name
            # code = irb.compile()
            moduleOp = new_v.body[0]
            for op in moduleOp.body:
                if isinstance(op, ops.FuncOp):
                    op.func_name = func_name
            code = new_v.dump()
            # start = time.time()
            lowering_result = lowering.lower_dialect_with_jit(code, target, None, new_flags, func_name, arg_vals, new_v.return_type)
            # end = time.time()
            # print("Time for JIT", end-start)
            return lowering_result

        return wrapper
    return innerfunc
