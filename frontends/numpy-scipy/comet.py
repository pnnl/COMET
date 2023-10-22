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

from _ast import Assign, Attribute, AugAssign, BinOp, Call, Constant, Expr
import ast
import inspect
from typing import Any
# import comet
import numpy as np
import scipy as sp
from MLIRGen import lowering
from MLIRGen import builders
from MLIRGen.types import *
#import time

def get_format(A):
    if not sp.sparse.issparse(A):
        return DENSE
    elif A.format == 'csr':
        return CSR
    elif A.format == 'coo':
        return COO
    elif A.format == 'csc':
        return CSC
    else:
        raise RuntimeError('Unsupported sparse format')





class NewVisitor(ast.NodeVisitor):

    def __init__(self,inputs):
        self.no_assign = False
        self.tsemantics = {}
        self.isemantics = {}
        self.tsymbols = {"comet": None}
        self.isymbols = {}
        self.tcurr = 0
        self.icurr = 0
        self.ops = []
        self.iLabelsToVals = {}
        self.valsToILabels = {}
        self.declarations = []
        self.uniqueLabels = []
        self.in_args = []
        self.returns = []
        self.mask = None
        self.need_opt_comp_workspace = False
        
        self.inputs = inputs
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


    def visit_FunctionDef(self, node):
        for i, arg in enumerate(node.args.args):
            self.tsymbols[arg.arg] = self.tcurr
            labels = []
            format = get_format(self.inputs[i])
            self.declarations.append(('d', 'T', 'i', i, self.tcurr))
            self.tsemantics[self.tcurr] = {
                'shape': list(self.inputs[i].shape),
                'format': format
                }
            self.in_args.append(self.tcurr)
            self.tcurr += 1
        
        for stmt in node.body:
            NewVisitor.visit(self, stmt)
    
    def visit_Assign(self, node): # We do not support multiple targets currently

        if isinstance(node.targets[0], ast.Subscript): 
            id = node.targets[0].value.id
            mask = node.targets[0].slice
            self.mask = mask
            v = NewVisitor.visit(self, node.value)
            self.tsymbols[id] = v
        else:
            # vals = [NewVisitor.visit(self, node.value)]
            # for l,v in zip(node.targets, vals):
            #     self.tsymbols[l.id] = v

            id = node.targets[0].id
            # if id in self.tsymbols:
            #     self.no_assign = True

            v = NewVisitor.visit(self, node.value)
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
                self.ops.append(("=", v, self.tsymbols[id], beta))
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.Call) and node.value.func.attr == "transpose" :
            v = self.visit_Call(node.value)
        
            if no_assign:
                self.ops.append(("=", v, self.tsymbols[id], beta))
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            v = self.visit_BinOp(node.value)
            if no_assign:
                self.ops.append(("=", v, self.tsymbols[id], beta))
            else:
                self.tsymbols[id] = v
        # Not a special case. Just add the result to the LHS
        else:
            v = NewVisitor.visit(self, node.value)
            res = self.create_binOp(node, [self.tsymbols[id], v], no_assign)
            self.tsymbols[id] = res


    def visit_Call(self, node: Call) -> Any:
        obj = None
        obj = NewVisitor.visit(self,node.func)
        if obj != None:
            return self.visit_Method_Call(node, obj) 
        else:
            if node.func.attr == "einsum": # Handle comet.einsum
                return self.visit_Einsum_Call(node)


    def visit_Attribute(self, node: Attribute) -> Any:
        return NewVisitor.visit(self, node.value)

    def visit_Name(self, node: ast.Name):
        return self.tsymbols[node.id]

    def visit_Constant(self, node: Constant) -> Any:
        out_id = self.tcurr
        self.declarations.append(('d', 'v', 'l', out_id))
        self.tcurr +=1
        return out_id
    
    def create_binOp(self, node, operands, no_assign):
        out_id = self.tcurr
        op0_sems = self.tsemantics[operands[0]]
        op1_sems = self.tsemantics[operands[1]]
        if not isinstance(node.op, ast.MatMult):
            
            if 'labels' not in op0_sems:
                labels = []
                if 'labels' not in op1_sems:
                    for d in op0_sems['shape']:
                        self.iLabelsToVals[self.icurr] = (d, op0_sems['format'] or op1_sems['format'])
                        labels.append(self.icurr)
                        self.icurr +=1
                    op1_sems['labels'] = labels
                else:
                    labels = op1_sems['labels']
                op0_sems['labels'] = labels
            else:
                if 'labels' not in op1_sems:
                    op1_sems['labels'] = op0_sems['labels']
            format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
            if self.tsemantics[operands[0]]['format'] != DENSE:
                op_semantics = self.tsemantics[operands[0]]
                self.tsemantics[operands[1]]['labels'] = op_semantics['labels']
            else:
                op_semantics = self.tsemantics[operands[1]]
                if self.tsemantics[operands[1]]['format'] != DENSE:
                    self.tsemantics[operands[0]]['labels'] = op_semantics['labels']

            s = 'a'
            indices = "".join(chr(ord(s)+i) for i in range(len(op_semantics['labels'])))
        self.need_opt_comp_workspace = op0_sems['format'] == CSR and op1_sems['format'] == CSR
        if isinstance(node.op, ast.Add):
            self.ops.append(("+", operands, indices +','+indices+'->'+indices, self.tcurr))

            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'labels': op_semantics['labels'], 'format': format}
            # if not no_assign:
            self.declarations.append(('d', 'T', 'l', self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assign))
        elif isinstance(node.op, ast.Sub):
            self.ops.append(("-", operands, indices+','+indices+'->'+indices, self.tcurr))
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'labels': op_semantics['labels'], 'format': format}
            # if not no_assign:
            self.declarations.append(('d', 'T', 'l', self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assign))
        elif isinstance(node.op, ast.Mult):
            self.ops.append(("*", operands, indices+','+indices+'->'+indices, self.tcurr, None))
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'labels': op_semantics['labels'], 'format': format}
            # if not no_assign:
            self.declarations.append(('d', 'T', 'l', self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assign))
        elif isinstance(node.op, ast.MatMult):
            mask = (None,None)
            if  self.mask != None:
                mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)
                mask_sems = self.tsemantics[mask[0]]
                if 'labels' not in mask_sems:
                    labels = []
                    for d in mask_sems['shape']:
                        self.iLabelsToVals[self.icurr] = (d, mask_sems['format'])
                        labels.append(self.icurr)
                        self.icurr += 1
                    mask_sems['labels'] = labels
                self.mask = None

                
            op1 = self.tsemantics[operands[0]]
            op2 = self.tsemantics[operands[1]]
            if len(op2['shape']) > 2 or len(op1['shape'])> 2:
                raise RuntimeError('')
            else:
                labels1 = []
                labels2 = []
                if 'labels' not in op1:
                    for d in op1['shape'][:-1]:
                        self.iLabelsToVals[self.icurr] = (d, op1['format'])
                        labels1.append(self.icurr)
                        self.icurr +=1
                    if 'labels' not in op2:
                        for d in op2['shape'][1:]:
                            self.iLabelsToVals[self.icurr] = (d, op2['format'])
                            labels2.append(self.icurr)
                            self.icurr +=1

                        self.iLabelsToVals[self.icurr] = (op2['shape'][0], op1['format'] or op2['format'])
                        op2['labels'] = [self.icurr] + labels2
                        self.icurr +=1
                    labels1.append(op2['labels'][0])
                    op1['labels'] = labels1
                else:
                    if 'labels' not in op2:
                        for d in op2['shape'][1:]:
                            self.iLabelsToVals[self.icurr] = (d, op2['format'])
                            labels2.append(self.icurr)
                            self.icurr +=1
                        op2['labels'] = [op1['labels'][-1]] + labels2
                        if op2['format']  != DENSE:
                            self.iLabelsToVals[op1['labels'][-1]] = (op2['shape'][0], op2['format'])



                if len(op1['shape']) == 2 and len(op2['shape']) == 2:
                    if op1['format'] != DENSE:
                        op2['labels'][0] = op1['labels'][1]
                    elif op2['format'] != DENSE:
                        op1['labels'][1] = op2['labels'][0]
                    shape = [op1['shape'][0], op2['shape'][-1]]
                    labels = [op1['labels'][0], op2['labels'][-1]]
                    indices = 'ab,bc->ac'
                elif len(op1['shape']) == 2 and len(op2['shape']) == 1:
                    if op1['format'] != DENSE:
                        op2['labels'][0] = op1['labels'][1]
                    elif op2['format'] != DENSE:
                        op1['labels'][1] = op2['labels'][0]
                    shape = [op1['shape'][0]]
                    labels = [op1['labels'][0]]
                    indices = 'ab,b->a'
                elif len(op1['shape']) == 1 and len(op2['shape']) == 2:
                    if op1['format'] != DENSE:
                        op2['labels'][0] = op1['labels'][0]
                    elif op2['format'] != DENSE:
                        op1['labels'][0] = op2['labels'][0]
                    shape = [op2['shape'][1]]
                    labels = [op2['labels'][-1]]
                    indices = 'a,ab->b'
                # elif len(op1['shape']) == 1 and len(op2['shape']) == 1:
                #     shape = [1,0]
            
            # self.ops.append(("c", operands, iLabels, self.tcurr, mask, mask_type, semiring))
            self.ops.append(("c", operands, indices, self.tcurr, mask[0], mask[1], None, no_assign))
            format = self.sp_matmult_conversions[op1['format']][op2['format']]
            self.tsemantics[self.tcurr] = {'shape': shape, 'labels': labels, 'format': format}
            if not no_assign:
                self.declarations.append(('d', 'T', 'l', self.tcurr))
                self.ops.append(("=", self.tcurr, self.tcurr, no_assign))

        self.tcurr +=1
        return out_id

    def visit_BinOp(self, node: BinOp) -> Any:
        no_assign = self.no_assign
        self.no_assign = False
        operands = [NewVisitor.visit(self, node.left), NewVisitor.visit(self, node.right)]
        return self.create_binOp(node, operands, no_assign)

    def visit_Method_Call(self, node: Call, obj):
        no_assign = self.no_assign
        self.no_assign = False
        # operands = []
        out_id = self.tcurr
        op_semantics = self.tsemantics[obj]
        format = op_semantics['format']
        if 'labels' not in op_semantics:
            labels = []
            for d in op_semantics['shape']:
                self.iLabelsToVals[self.icurr] = (d, format)
                labels.append(self.icurr)
                self.icurr += 1
            op_semantics['labels'] = labels
        if node.func.attr == "transpose":
            out_format = self.sp_mattr_conversions[op_semantics['format']]
            self.tsemantics[out_id] = {'shape': op_semantics['shape'][::-1], 'labels': op_semantics['labels'][::-1], 'format': out_format}
            self.ops.append(("t", [obj], 'ij->ji', out_id))
            # if not no_assign:
            self.declarations.append(('d', 'T', 'l', out_id))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assign))
        elif node.func.attr == "sum":
            self.tsemantics[out_id] = {'shape': [1,], 'format': DENSE, 'labels': []}
            self.ops.append(("s", [obj], out_id))
            # if not no_assign:
            self.declarations.append(('d', 'v', 'l', out_id))
        elif node.func.attr == "multiply":
            op1 = NewVisitor.visit(self, node.args[0])
            op1_sems = self.tsemantics[op1]
            if 'labels' not in op1_sems:
                op1_sems['labels'] = op_semantics['labels']
            if self.tsemantics[obj]['format'] != DENSE:
                op_semantics = self.tsemantics[obj]
                self.tsemantics[op1]['labels'] = op_semantics['labels']
            else:
                op_semantics = self.tsemantics[op1]
                if self.tsemantics[op1]['format'] != DENSE:
                    self.tsemantics[obj]['labels'] = op_semantics['labels']
            s = 'a'
            indices = "".join(chr(ord(s)+i) for i in range(len(op_semantics['labels'])))
            self.ops.append(("*", [obj, op1], indices+','+indices+'->'+indices, self.tcurr, None, no_assign))
            format = self.sp_elw_mult_conversions[op_semantics['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'labels': op_semantics['labels'], 'format': format}
            # if not no_assign:
            self.declarations.append(('d', 'T', 'l', self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assign))
        self.tcurr +=1

        return out_id

    def visit_Return(self, node):
        obj = NewVisitor.visit(self, node.value)
        self.returns.append(obj)
        if self.tsemantics[obj]['format'] == DENSE:
            self.in_args.append(obj)
        self.ops.append(("p", [obj]))


    def visit_Einsum_Call(self, node: Call):
        no_assing =  self.no_assign
        self.no_assign = False
        out_id = self.tcurr
        mask = (None, None)
        # mask_type = "none"
        semiring = None
        if  self.mask != None:
            mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)
            mask_sems = self.tsemantics[mask[0]]
            if 'labels' not in mask_sems:
                labels = []
                for d in mask_sems['shape']:
                    self.iLabelsToVals[self.icurr] = (d, mask_sems['format'])
                    labels.append(self.icurr)
                    self.icurr += 1
                mask_sems['labels'] = labels
            self.mask = None
        iLabels = node.args[0].value
        ops, res = iLabels.split('->')
        ops = ops.split(',')
        res = list(res)
        if len(node.keywords) > 0:
            for key in node.keywords:
                if key.arg == 'mask':
                    mask = NewVisitor.visit(self, key.value)
                    mask_sems = self.tsemantics[mask]
                    if 'labels' not in mask_sems:
                        labels = []
                        for d in mask_sems['shape']:
                            self.iLabelsToVals[self.icurr] = (d, mask_sems['format'])
                            labels.append(self.icurr)
                            self.icurr += 1
                        mask_sems['labels'] = labels
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
            
            if 'labels' not in op0_sems:
                labels = []
                if 'labels' not in op1_sems:
                    for d in op0_sems['shape']:
                        self.iLabelsToVals[self.icurr] = (d, op0_sems['format'] or op1_sems['format'])
                        labels.append(self.icurr)
                        self.icurr +=1
                    op1_sems['labels'] = labels
                else:
                    labels = op1_sems['labels']
                op0_sems['labels'] = labels
            else:
                if 'labels' not in op1_sems:
                    op1_sems['labels'] = op0_sems['labels']
            format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
            if self.tsemantics[operands[0]]['format'] != DENSE:
                op_semantics = self.tsemantics[operands[0]]
                self.tsemantics[operands[1]]['labels'] = op_semantics['labels']
            else:
                op_semantics = self.tsemantics[operands[1]]
                if self.tsemantics[operands[1]]['format'] != DENSE:
                    self.tsemantics[operands[0]]['labels'] = op_semantics['labels']

            s = 'a'
            indices = "".join(chr(ord(s)+i) for i in range(len(op_semantics['labels'])))
            self.ops.append(("*", operands, indices+','+indices+'->'+indices, self.tcurr, semiring))
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'labels': op_semantics['labels'], 'format': format}
            self.declarations.append(('d', 'T', 'l', self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assing))

            self.tcurr += 1

            return out_id
        if len(operands) > 1:
            i = 0
            opl0, opl1 = list(ops[0]), list(ops[1])
            if 'labels' not in self.tsemantics[operands[0]]:
                if 'labels' not in self.tsemantics[operands[1]]:
                    labels = []
                    for d in self.tsemantics[operands[1]]['shape']:
                        labels.append(self.icurr)
                        self.iLabelsToVals[self.icurr] = (d, self.tsemantics[operands[1]]['format'])
                        self.icurr += 1
                    self.tsemantics[operands[1]]['labels'] = labels

                labels = []
                for i in range(len(opl0)):
                    j = 0
                    found = False
                    for j in range(len(opl1)):
                        if opl0[i] == opl1[j]:
                            found = True
                            if self.tsemantics[operands[0]]['format'] != DENSE:
                                self.iLabelsToVals[self.tsemantics[operands[1]]['labels'][j]] = (self.tsemantics[operands[0]]['shape'][j], self.tsemantics[operands[0]]['format'])
                            labels.append(self.tsemantics[operands[1]]['labels'][j])
                    if not found:
                        labels.append(self.icurr)
                        self.iLabelsToVals[self.icurr] = (self.tsemantics[operands[0]]['shape'][i], self.tsemantics[operands[0]]['format'])
                        self.icurr += 1
                self.tsemantics[operands[0]]['labels'] = labels
            else:
                if 'labels' not in self.tsemantics[operands[1]]:
                    labels = []
                    for j in range(len(opl1)):
                        i = 0
                        found = False
                        for i in range(len(opl0)):
                            if opl0[i] == opl1[j]:
                                found = True
                                if self.tsemantics[operands[1]]['format'] != DENSE:
                                    self.iLabelsToVals[self.tsemantics[operands[0]]['labels'][i]] = (self.tsemantics[operands[1]]['shape'][j], self.tsemantics[operands[1]]['format'])
                                labels.append(self.tsemantics[operands[0]]['labels'][i])
                        if not found:
                            labels.append(self.icurr)
                            self.iLabelsToVals[self.icurr] = (self.tsemantics[operands[1]]['shape'][j], self.tsemantics[operands[1]]['format'])
                            self.icurr += 1
                    self.tsemantics[operands[1]]['labels'] = labels

            if self.tsemantics[operands[0]]['format'] != DENSE and  self.tsemantics[operands[1]]['format'] == DENSE:
                while i < len(opl0):
                    j = 0
                    while j < len(opl1):
                        if opl0[i] == opl1[j]:
                            self.tsemantics[operands[1]]['labels'][j] = self.tsemantics[operands[0]]['labels'][i]
                            break
                        else:
                            j +=1
                    i += 1
                    
            elif self.tsemantics[operands[0]]['format'] == DENSE and  self.tsemantics[operands[1]]['format'] != DENSE:
                while i < len(opl0):
                    j = 0
                    while j < len(opl1):
                        if opl0[i] == opl1[j]:
                            self.tsemantics[operands[0]]['labels'][i] = self.tsemantics[operands[1]]['labels'][j]
                            break
                        else:
                            j +=1
                    i += 1
        if not no_assing or len(operands) == 1:
            self.declarations.append(('d', 'T', 'l', self.tcurr))
        labels = []

        if len(operands) == 1:
            if 'labels'  not in self.tsemantics[operands[0]] :
                labels = []
                for d in self.tsemantics[operands[0]]['shape']:
                    labels.append(self.icurr)
                    self.iLabelsToVals[self.icurr] = (d, self.tsemantics[operands[0]]['format'])
                    self.icurr +=1
                self.tsemantics[operands[0]]['labels'] = labels


        temp = []
        for _ in range(len(ops)):
            temp.append([])
        order = []
        for l in res:
            found = False
            for i,op in enumerate(ops):
                if found:
                    break
                for j,lop in enumerate(op):
                    if lop == l:
                        order.append((i,j))
                        temp[i].append(j)
                        found = True
                        break
        shape = []
        labels = []
        for (t, pos) in order:
            oshape = self.tsemantics[operands[t]]['shape']
            olabels = self.tsemantics[operands[t]]['labels']
            shape.append(oshape[pos])
            labels.append(olabels[pos])
        if len(operands) == 1:
            format = self.sp_mattr_conversions[self.tsemantics[operands[0]]['format']]
            self.ops.append(("t", operands, iLabels, self.tcurr))
            self.ops.append(("=", self.tcurr, self.tcurr, no_assing))

        else:
            
            format = self.tsemantics[operands[0]]['format']
            if format != DENSE:
                 self.need_opt_comp_workspace = True
            for op in operands[1:]:
                format = self.sp_matmult_conversions[format][self.tsemantics[op]['format']]
                if format != DENSE:
                    self.need_opt_comp_workspace = True
            self.ops.append(("c", operands, iLabels, self.tcurr, mask[0], mask[1], semiring, no_assing))
            if not no_assing:
                self.ops.append(("=", self.tcurr, self.tcurr, no_assing))
        self.tsemantics[self.tcurr] = {'shape': shape, 'labels': labels, 'format': format}
        self.tcurr += 1

        return out_id


#Wrapper function. The input function (in the form of an object) is passed an arguement to this function.
def compile(flags, with_jit=True):
    def innerfunc(func):

        def wrapper(*pos_args, **kwargs):
            func_str = ast.parse(inspect.getsource(func))
            parsed_func = ast.parse(func_str)
            func_def = parsed_func.body[0]
            v = NewVisitor([*pos_args])
            v.visit(parsed_func)
            in_types = []
            for arg in v.in_args:
                if len(v.tsemantics[arg]['labels']) == 0:
                    in_types.append(("%t"+str(arg), "tensor<1xf64>"))
                else:
                    in_types.append(("%t"+str(arg), "tensor<{}xf64>".format("x".join(str(v.iLabelsToVals[s][0]) if v.iLabelsToVals[s][1] == DENSE else '?' for s in v.tsemantics[arg]['labels']))))
            irb = builders.MLIRFunctionBuilder(
                func_def.name,
                input_types=in_types,
                return_types=[],
            ) 

            vars = []
            sparse_tensors = []
            dense_tensors = []
            sILabels = {}
            dILabels = {}

            v.uniqueLabels = set(v.uniqueLabels)
            c0 = "%c0 = arith.constant 0 : index"
            c1 = "%c1 = arith.constant 1 : index"
            irb.add_statement(c0)
            irb.add_statement(c1)
            valMap = {}
            for i in v.iLabelsToVals:
                val, t = v.iLabelsToVals[i]
                if t == DENSE:
                    if val not in valMap:
                        irb.add_statement('%c{} = arith.constant {} : index'.format(val, val))
                        valMap[val] = "in"
                    irb.add_statement('%i{} = "ta.static_index_label"(%c0, %c{}, %c1) : (index,index,index) -> !ta.range'.format(i, val))
                else:
                    irb.add_statement('%i{} = "ta.dynamic_index_label"(%c0, %c1) : (index,index) -> !ta.range'.format(i))


            label_map = v.iLabelsToVals
            dense_tensors = []
            sp_tensors = []
            for dec in v.declarations:
                
                if dec[1] == 'T':
                    t = builders.Tensor_Decl_Builder(dec[-1], v.tsemantics[dec[-1]]['labels'], v.tsemantics[dec[-1]]['labels'], v.tsemantics[dec[-1]]['format'], 'f64', label_map, dec[2] == 'i')
                    if v.tsemantics[dec[-1]]['format'] == DENSE:
                        dense_tensors.append(t)
                    else:
                        irb.add_statement(t.build_tensor())

            for t in dense_tensors:
                irb.add_statement(t.build_tensor())

            for op in v.ops:
                
                if op[0] == 'c':
                    if op[4] != None:
                        irb.add_statement(builders.ArithOp_Builder(op[3], op[1], op[2], [v.tsemantics[t]['format'] for t in op[1]] +  [v.tsemantics[op[3]]['format']], [v.tsemantics[t]['labels'] for t in op[1]] +  [v.tsemantics[op[3]]['labels']], op[0], label_map, op[4], op[5], v.tsemantics[op[4]]['labels'], op[6], op[7]  ).build_op())
                    else:
                        irb.add_statement(builders.ArithOp_Builder(op[3], op[1], op[2], [v.tsemantics[t]['format'] for t in op[1]] +  [v.tsemantics[op[3]]['format']], [v.tsemantics[t]['labels'] for t in op[1]] +  [v.tsemantics[op[3]]['labels']], op[0], label_map, op[4], op[5], None, op[6], op[7]  ).build_op())
                elif op[0] == 's':
                    irb.add_statement(builders.TensorSumBuilder(op[-1], op[1], [v.tsemantics[t]['labels'] for t in op[1]], label_map).build_op())
                elif op[0] == 'p':
                    irb.add_statement(builders.PrintBuilder(op[1], [v.tsemantics[t]['labels'] for t in op[1]], "f64", label_map).build_op())
                elif op[0] == '*':
                    irb.add_statement(builders.ArithOp_Builder(op[3], op[1], op[2], [v.tsemantics[t]['format'] for t in op[1]] +  [v.tsemantics[op[3]]['format']], [v.tsemantics[t]['labels'] for t in op[1]] +  [v.tsemantics[op[3]]['labels']], op[0], label_map, None, "none", None, op[4] ).build_op())
                elif op[0] == '=':
                    irb.add_statement(builders.SetOp_Builder(op[1], op[2], [v.tsemantics[op[1]]['labels']], label_map, op[3] ).build_op())
                else:
                    irb.add_statement(builders.ArithOp_Builder(op[-1], op[1], op[2], [v.tsemantics[t]['format'] for t in op[1]] +  [v.tsemantics[op[-1]]['format']], [v.tsemantics[t]['labels'] for t in op[1]] +  [v.tsemantics[op[-1]]['labels']], op[0], label_map ).build_op())
            irb.add_statement("return")

            outputs = []
            ret = v.tsemantics[v.returns[0]]
            format = ret['format']
            if format == DENSE:
                outputs.append(np.empty(ret['shape']))
            elif format == CSR:
                outputs.append(sp.sparse.csr_array([]))
            elif format == COO:
                outputs.append(sp.sparse.coo_array([]))
            elif format == CSC:
                outputs.append(sp.sparse.csc_array([]))

            arg_vals = v.inputs
            new_flags = flags
            if v.need_opt_comp_workspace:
                if new_flags:
                    new_flags = new_flags + ' --opt-comp-workspace'
                else:
                    new_flags = ' --opt-comp-workspace'
            code = irb.compile()
            # start = time.time()
            lowering_result = lowering.lower_dialect_with_jit(code,None, new_flags,func_def.name, arg_vals, outputs)
            # end = time.time()
            # print("Time for JIT", end-start)
            return lowering_result

        return wrapper
    return innerfunc