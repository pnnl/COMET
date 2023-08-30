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

import ast
from MLIRGen.utils import HashTable,oprs_info
from MLIRGen import lowering
from MLIRGen import PyMLIRGen
import inspect,sys
import numpy as np
import scipy as scp

#Node Visitor class to visit every node in the function ast
    
class _AnalysisNodeVisitor(ast.NodeVisitor):

    #Visit nodes of types: x = ...
    def visit_Assign(self,node):
        if(isinstance(node.targets[0],ast.Name)):
            target = node.targets[0].id
            if(isinstance(node.value, ast.Call)):
                if(ast.unparse(node.value.func) == "comet.einsum"):
                    _AnalysisNodeVisitor.visit_einsum_Call(self,node.value,target, '=')
                    numpy_array_info.set_val(target, 0.0)
                 
                elif(ast.unparse(node.value.func) == "comet.copy"):
                    copy_call_args = node.value.args[0]
                    if(isinstance(copy_call_args,ast.Call)):
                        if(ast.unparse(copy_call_args.func) == "comet.einsum"):
                            _AnalysisNodeVisitor.visit_einsum_Call(self,copy_call_args,target, '=')
                            numpy_array_info.set_val(target, 0.0)

                elif(ast.unparse(node.value.func) == "comet.multiply"):
                    _AnalysisNodeVisitor.visit_Call_elewise_mul(self,node.value,target, '=')
                    #numpy_array_info.set_val(target,0.0)

                elif(ast.unparse(node.value.func) == "np.ones"):
                    _AnalysisNodeVisitor.visit_Call_numpy(self,node.value,target)
                    numpy_array_info.set_val(target, 1.0)
                 
                elif(ast.unparse(node.value.func) == "np.zeros"):
                    _AnalysisNodeVisitor.visit_Call_numpy(self,node.value,target)
                    numpy_array_info.set_val(target, 0.0)

                elif(ast.unparse(node.value.func) == "np.full"):
                    _AnalysisNodeVisitor.visit_Call_numpy(self,node.value, target)
                
                elif("transpose" in ast.unparse(node.value.func)):
                    _AnalysisNodeVisitor.visit_einsum_Call(self,node.value,target, '=')
                    numpy_array_info.set_val(target, 0.0)

                elif("sum" in ast.unparse(node.value.func)):
                    _AnalysisNodeVisitor.visit_sum_Call(self,node.value,target, '=')
                    if(target not in list_numpy_arrays):
                        list_numpy_arrays.append(target)
                        numpy_array_info.set_val(target, None)


            #Handles the case where target = -comet.einsum()
            elif(isinstance(node.value, ast.UnaryOp)):
                _AnalysisNodeVisitor.visit_UnaryOpr(self,node.value,target) 

             #Handles the case where target = x op y
            elif(isinstance(node.value, ast.BinOp)):
                _AnalysisNodeVisitor.visit_BinaryOp(self,node.value,target) 

            if(target not in list_numpy_arrays):
                list_numpy_arrays.append(target)
                numpy_array_info.set_val(target, 0.0)

    def visit_sum_Call(self, node, target, op):
        opr_no = "opr" + str(node.lineno)
        input_arrays = [ast.unparse(node.func.value)]

        input_dim_labels = []
        labels_defined_for = []
        for i in range(len(input_arrays)):
            lbl = numpy_array_info.get_dims_labels(input_arrays[i])
            if lbl not in "No record found": # First occurence of this tensor before any einsum
                labels_defined_for.append(i)

        labels_known = None
        if len(labels_defined_for) != len(input_arrays):
            for i in range(len(input_arrays)):
                if i not in labels_defined_for:

                    if labels_known == None:
                        dims = numpy_array_info.get_dims(input_arrays[i])
                        dims_lbls = []
                        for d,dval in enumerate(dims):
                            label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(i)] = dval
                            dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(i))
                        numpy_array_info.set_dims_labels(input_arrays[i] ,dims_lbls)
                        labels_known = dims_lbls
                    else:
                        numpy_array_info.set_dims_labels(input_arrays[i] ,labels_known)
        
        
        numpy_array_info.set_format(target, "Dense")  

        oprs_info(opr_no,target,input_arrays, [], [],"tensor_sum", "sum")
        list_operations.append(opr_no)
        

    #Visit nodes of types: target += comet.einsum() and target -= ... 
    def visit_AugAssign(self, node):
        if(isinstance(node.target,ast.Name)):
            target = node.target.id
            if(isinstance(node.value, ast.Call)):
                if(ast.unparse(node.value.func) == "comet.einsum"):
                    if(isinstance(node.op, ast.Add)):
                        _AnalysisNodeVisitor.visit_einsum_Call(self,node.value,target, '+=')
                    elif(isinstance(node.op, ast.Sub)):
                        _AnalysisNodeVisitor.visit_einsum_Call(self,node.value,target, '-=')
                    
                if(target not in list_numpy_arrays):
                    list_numpy_arrays.append(target)
                    numpy_array_info.set_val(target, 0.0)

            elif(isinstance(node.value, ast.UnaryOp)):
                _AnalysisNodeVisitor.visit_UnaryOpr(self,node.value,target)
            
            elif(isinstance(node.value, ast.Name)):
                opr_no = "op" + str(node.lineno)
                input_arrays = [target, node.value.id]
                
                dims_lbls = []
                target_dims_lbls = numpy_array_info.get_dims_labels(input_arrays[0])
                for a in input_arrays:
                    dims_lbls.append(numpy_array_info.get_dims_labels(a))
                if(isinstance(node.op, ast.Add)):    
                    oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_add", "+")
                elif(isinstance(node.op, ast.Sub)):
                    oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_sub", "-")

                list_operations.append(opr_no)
        
        return
       
    #Support for expression of the type: target += -comet.ensum('ai,ik->ak',A,B)
    def visit_UnaryOpr(self, node,target):
        if(isinstance(node.op, ast.USub)):
            _AnalysisNodeVisitor.visit_einsum_Call(self,node.operand,target, '-=')
            if(target not in list_numpy_arrays):
                list_numpy_arrays.append(target)
                numpy_array_info.set_val(target, 0.0)
        return
        
    #Handling Binary operations involving 
    # 1. only tensors: e.g. x[i,j] = y[i,j] + z[i,j]
    # 2. Tensor and einsum: e.g. x[i,j] = y[i,j] + comet.einsum(...)
    def visit_BinaryOp(self, node:ast.BinOp, target):
        opr_no = "opr" + str(node.lineno)
        input_arrays = []
        #Handles the case: x[i,j] = y[i,j] + z[i,j]
        if(isinstance(node.left, ast.Name) and 
            isinstance(node.right, ast.Name)):

            if(isinstance(node.op, ast.Mult)):
                self.visit_Call_elewise_mul(node, target,'=')
                return
            elif(isinstance(node.op, ast.MatMult)):
                self.visit_einsum_Call(node, target,'=')
                return

            input_arrays = [node.left.id, node.right.id]

            input_dim_labels = []
            labels_defined_for = []
            for i in range(len(input_arrays)):
                lbl = numpy_array_info.get_dims_labels(input_arrays[i])
                if lbl not in "No record found": # First occurence of this tensor before any einsum
                    labels_defined_for.append(i)

            labels_known = None

            if len(labels_defined_for) != len(input_arrays):
                if numpy_array_info.get_dims_labels(target) != "No record found":
                    labels_known = numpy_array_info.get_dims_labels(target)
                
                for i in range(len(input_arrays)):
                    if i not in labels_defined_for:

                        if labels_known == None:
                            dims = numpy_array_info.get_dims(input_arrays[i])
                            dims_lbls = []
                            for d,dval in enumerate(dims):
                                label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(i)] = dval
                                dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(i))
                            numpy_array_info.set_dims_labels(input_arrays[i] ,dims_lbls)
                            labels_known = dims_lbls
                        else:
                            numpy_array_info.set_dims_labels(input_arrays[i] ,labels_known)
            
            for i in range(len(input_arrays)):
                input_dim_labels.append(numpy_array_info.get_dims_labels(input_arrays[i]))
            
            output_dim_labels = input_dim_labels[0]
            numpy_array_info.set_dims(target,numpy_array_info.get_dims(input_arrays[0]))
            numpy_array_info.set_dims_labels(target,output_dim_labels)

            output_format = sp_elw_add_sub_conversions[numpy_array_info.get_format(input_arrays[0])][numpy_array_info.get_format(input_arrays[1])]
            numpy_array_info.set_format(target, output_format)  


            dims_lbls = []
            target_dims_lbls = numpy_array_info.get_dims_labels(input_arrays[0])
            for a in input_arrays:
                dims_lbls.append(numpy_array_info.get_dims_labels(a))
            if(isinstance(node.op, ast.Add)):
                oprs_info(opr_no,target,input_arrays, dims_lbls,target_dims_lbls,"tensor_add", "+")
            elif(isinstance(node.op, ast.Sub)):
                oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_sub", "-")
            
            target_dims = numpy_array_info.get_dims(ast.unparse(node.left))
        
        #Handles the case: x[i,j] = y[i,j] + comet.einsum(...)
        #P.S: The LHS here is a tensor(ast.Name)
        elif(isinstance(node.left, ast.Name) and
                isinstance(node.right, ast.Call)):
                count = 0
                result = "tmp" + str(node.lineno)
                opr_no = "opr" + str(node.lineno) + "_" +  str(count)
                _AnalysisNodeVisitor.visit_einsum_Call(self,node.right,result,"=")
                count +=1
                if(result not in list_numpy_arrays):
                    list_numpy_arrays.append(result)
                    numpy_array_info.set_val(result, 0.0)
                
                opr_no = "opr" + str(node.lineno) + "_" + str(count)
                input_arrays = [ast.unparse(node.left), result]   
                
                dims_lbls = []
                target_dims_lbls = numpy_array_info.get_dims_labels(input_arrays[0])
                for a in input_arrays:
                    dims_lbls.append(numpy_array_info.get_dims_labels(a))
                if(isinstance(node.op, ast.Add)): 
                    oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_add", "+")
                elif(isinstance(node.op, ast.Sub)):
                    oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_sub", "-")
                elif(isinstance(node.op, ast.MatMult)):
                    self.visit_einsum_Call(node, target,'=')
                    return
                
                target_dims = numpy_array_info.get_dims(ast.unparse(node.left))

        #Handles the case:  x[i,j] = -y[i,j] + comet.einsum(...)
        # P.S.: In this case the LHS is a Unary Operand (ast.UnaryOp) tensor      
        elif(isinstance(node.left, ast.UnaryOp) and
                isinstance(node.right, ast.Call)):
                count = 0
                result = "tmp" + str(node.lineno)
                opr_no = "opr" + str(node.lineno) + "_" +  str(count)
                _AnalysisNodeVisitor.visit_einsum_Call(self,node.right,result,"=")
                count +=1
                if(result not in list_numpy_arrays):
                    list_numpy_arrays.append(result)
                    numpy_array_info.set_val(result, 0.0)
                
                opr_no = "opr" + str(node.lineno) + "_" + str(count)
                dims_lbls = []
                target_dims_lbls = numpy_array_info.get_dims_labels(input_arrays[0])
                for a in input_arrays:
                    dims_lbls.append(numpy_array_info.get_dims_labels(a))
                if(isinstance(node.left.op, ast.USub)):
                    input_arrays = [result,ast.unparse(node.left.operand)]
                    oprs_info(opr_no,target,input_arrays,dims_lbls,target_dims_lbls,"tensor_sub", "-")
                
                target_dims = numpy_array_info.get_dims(ast.unparse(node.left.operand))


        #If the result of the arithmetic operation is not previously initialized
        # it's dimensions need to be determined. The dimensions of the target will be
        # the same as the operands of the arithmetic, since it is elementwise operation.     
        if(numpy_array_info.get_dims(target) == "No record found"): 
            numpy_array_info.set_dims(target,target_dims)

        list_operations.append(opr_no)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    #Handles the einsum function call : comet.einsum(...)
    #Support for contraction and transpose operations.
    def visit_einsum_Call(self, node, target, op):
        opr_no = "opr" + str(node.lineno)
        input_arrays = []
        transpose = False

        if isinstance(node, ast.BinOp): # einsum hidden as a @ operation
            input_arrays = [node.left.id,node.right.id] 
            labels_defined_for = []
            input_dim_lbls = []
            lbls0 = numpy_array_info.get_dims_labels(input_arrays[0])
            lbls1 = numpy_array_info.get_dims_labels(input_arrays[1])
            if len(numpy_array_info.get_dims(input_arrays[0])) > 2 or len(numpy_array_info.get_dims(input_arrays[1])) > 2:
                   raise RuntimeError(
                        "@ operator can only be used with arrays of rank <=2. Please use einsum for arrays of larger rank"
                    )
            if lbls0 == "No record found": # First occurence of this tensor before any einsum
                dims0 = numpy_array_info.get_dims(input_arrays[0])
                dims1 = numpy_array_info.get_dims(input_arrays[1])
                if lbls1 == "No record found": # First occurence of this tensor before any einsum
                    dims_lbls = []
                    for d,dval in enumerate(dims1):
                        label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(1)] = dval
                        dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(1))
                    numpy_array_info.set_dims_labels(input_arrays[1] ,dims_lbls)
                
                dims_lbls = []
                for i in range(len(dims0)-1):
                    dval = dims0[i]
                    d = i
                    label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(0)] = dval
                    dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(0))
                
                dims_lbls.append(numpy_array_info.get_dims_labels(input_arrays[1])[0])
                                 
                numpy_array_info.set_dims_labels(input_arrays[0] ,dims_lbls)
            else:
                if lbls1 == "No record found": # First occurence of this tensor before any einsum
                    dims1 = numpy_array_info.get_dims(input_arrays[1])
                    dims_lbls = []
                    dims_lbls.append(numpy_array_info.get_dims_labels(input_arrays[0])[-1])
                    
                    for i in range(1, len(dims1)):
                        dval = dims1[i]
                        d = i
                        label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(1)] = dval
                        dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(1))
                    
                    numpy_array_info.set_dims_labels(input_arrays[1] ,dims_lbls)
            output_dim_lbls = [x for x in numpy_array_info.get_dims_labels(input_arrays[0]) if x not in numpy_array_info.get_dims_labels(input_arrays[1])]
            output_dim_lbls += [x for x in numpy_array_info.get_dims_labels(input_arrays[1]) if x not in numpy_array_info.get_dims_labels(input_arrays[0])]
            opr_type = "contraction"
            list_operations.append(opr_no)

            for i in range(len(input_arrays)):
                input_dim_lbls.append(numpy_array_info.get_dims_labels(input_arrays[i]))

                
            oprs_info(opr_no,target,input_arrays,input_dim_lbls,output_dim_lbls,opr_type,op)

        else:
            if len(node.args)> 0:
                for arg in node.args:
                    if(isinstance(arg, ast.Name)):
                        input_arrays.append(arg.id)

                    #Analyze the mapping to get the dimensions of the output tensor
                    elif(isinstance(arg, ast.Constant)):
                        mapping = (arg.value).split("->")
                        #If two input (comma separated) dimension labels exist in the mapping, it is a tensor contraction
                        if("," in mapping[0]):
                            input_dim_lbls = mapping[0].split(",")
                            input1_dims = list(input_dim_lbls[0])
                            input2_dims = list(input_dim_lbls[1])
                            output_dim_lbls = list(mapping[1])
                            # input_dim_lbls = [input1_dims,input2_dims]
                            input_dim_lbls = [list(x) for x in input_dim_lbls]
                            opr_type = "contraction"
                            oprs_info(opr_no,target,input_arrays,input_dim_lbls,output_dim_lbls,opr_type,op)
                            list_operations.append(opr_no)
                        #Else it is a tensor transpose
                        else:
                            transpose = True
                            input_dims = list(mapping[0])
                            output_dim_lbls = list(mapping[1])
                            input_dim_lbls = [input_dims]
                            opr_type = "transpose"
                            oprs_info(opr_no,target,input_arrays,input_dim_lbls,output_dim_lbls,opr_type,op)
                            list_operations.append(opr_no)
            
                #Map the labels in the einsum input syntax to their corresponding values stored in the hash table
                for i,j in zip(range(len(input_arrays)), range(len(input_dim_lbls))):  
                    dim_values = numpy_array_info.get_dims(input_arrays[i])
                    dim_lbls = input_dim_lbls[j]
                    for key,value in zip(dim_lbls,dim_values):
                        label_id_map[key] = value #[TODO] What if the same index is used for different sizes in different contractions?
                            

                    numpy_array_info.set_dims_labels(input_arrays[i],dim_lbls)
            
            elif isinstance(node.func, ast.Attribute) and node.func.attr == 'transpose':
                transpose = True
                input_arrays = [ast.unparse(node.func.value)]
                labels_defined_for = []
                input_dim_lbls = []
                lbls0 = numpy_array_info.get_dims_labels(input_arrays[0])
                if len(numpy_array_info.get_dims(input_arrays[0])) > 2:
                    raise RuntimeError(
                            "method transpose can only be used with arrays of rank 2. Please use einsum for arrays of larger rank"
                        )
                if lbls0 == "No record found": # First occurence of this tensor before any einsum
                    dims0 = numpy_array_info.get_dims(input_arrays[0])
                    dims_lbls = []
                    for i in range(len(dims0)):
                        dval = dims0[i]
                        d = i
                        label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(0)] = dval
                        dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(0))
                    
                    numpy_array_info.set_dims_labels(input_arrays[0] ,dims_lbls)

                output_dim_lbls = list(reversed(numpy_array_info.get_dims_labels(input_arrays[0])))
                opr_type = "transpose"
                list_operations.append(opr_no)

                for i in range(len(input_arrays)):
                    input_dim_lbls.append(numpy_array_info.get_dims_labels(input_arrays[i]))

                    
                oprs_info(opr_no,target,input_arrays,input_dim_lbls,output_dim_lbls,opr_type,op)
                # print(ast.unparse(node.func.value))
            else:
                raise RuntimeError("Unexpected error")

        if transpose:
            # Use lookup table to extract the format of the output based on the input
            output_format = sp_mattr_conversions[numpy_array_info.get_format(input_arrays[0])]
        else: # contraction
            if  numpy_array_info.get_format(input_arrays[0]) != "Dense" or numpy_array_info.get_format(input_arrays[1]) != "Dense":
                global need_opt_comp_workspace 
                need_opt_comp_workspace = True

            # Use lookup table to extract the format of the output based on the input
            output_format = sp_matmult_conversions[numpy_array_info.get_format(input_arrays[0])][numpy_array_info.get_format(input_arrays[1])]
            if len(input_arrays) > 2:
                for a in input_arrays[2:]:
                    output_format = sp_matmult_conversions[output_format][numpy_array_info.get_format(a)]

        output_dim_vals = []
        for lbl in output_dim_lbls:
            if output_format == "Dense":
                output_dim_vals.append(label_id_map[lbl])
            else:
                output_dim_vals.append('?')
        numpy_array_info.set_format(target,output_format)
        numpy_array_info.set_dims(target,output_dim_vals)
        numpy_array_info.set_dims_labels(target,output_dim_lbls)



        return

    #Handles calls to numpy.ones(), numpy.zeros() and numpy.full()
    #within the target function. (Rarely required)
    def visit_Call_numpy(self, node, target):
        list_args = []
        for arg in node.args:
            if(isinstance(arg, ast.Constant)):
                numpy_array_info.set_val(target,float(arg.value))

            elif(isinstance(arg, ast.Tuple)):
                for element in arg.elts:
                    if(isinstance(element, ast.Constant)):
                        list_args.append(element.value)
                numpy_array_info.set_dims(target, list_args)
            else:
                raise TypeError("Unsupported operation")

        return

    def visit_Call_elewise_mul(self, node, target,op):
        opr_no = "opr" + str(node.lineno)
        input_arrays = []


        if(isinstance(node, ast.BinOp)): # Called from a A * B operation
            input_arrays = [node.left.id,node.right.id] 
        else:
            for arg in node.args:
                if(isinstance(arg, ast.Name)):
                    input_arrays.append(arg.id)

        input_dim_labels = []
        labels_defined_for = []
        for i in range(len(input_arrays)):
            lbl = numpy_array_info.get_dims_labels(input_arrays[i])
            if lbl not in "No record found": # First occurence of this tensor before any einsum
                labels_defined_for.append(i)

        labels_known = None

        if len(labels_defined_for) != len(input_arrays):
            if numpy_array_info.get_dims_labels(target) != "No record found":
                labels_known = numpy_array_info.get_dims_labels(target)
            
            for i in range(len(input_arrays)):
                if i not in labels_defined_for:

                    if labels_known == None:
                        dims = numpy_array_info.get_dims(input_arrays[i])
                        dims_lbls = []
                        for d,dval in enumerate(dims):
                            label_id_map['idx'+str(node.lineno)+'d'+str(d)+'i'+str(i)] = dval
                            dims_lbls.append('idx'+str(node.lineno)+'d'+str(d)+'i'+str(i))
                        numpy_array_info.set_dims_labels(input_arrays[i] ,dims_lbls)
                        labels_known = dims_lbls
                    else:
                        numpy_array_info.set_dims_labels(input_arrays[i] ,labels_known)
        
        for i in range(len(input_arrays)):
            input_dim_labels.append(numpy_array_info.get_dims_labels(input_arrays[i]))
        
        output_dim_labels = input_dim_labels[0]
        numpy_array_info.set_dims(target,numpy_array_info.get_dims(input_arrays[0]))
        numpy_array_info.set_dims_labels(target,output_dim_labels)
        
        # Currently the format of the output tensor is the same as the one of the first operand 
        # [TODO] See what scipy returns

        output_format = sp_elw_mult_conversions[numpy_array_info.get_format(input_arrays[0])][numpy_array_info.get_format(input_arrays[1])]
        numpy_array_info.set_format(target, output_format)  
        oprs_info(opr_no,target,input_arrays,input_dim_labels,output_dim_labels,"elewise_mult",op)
        list_operations.append(opr_no)

    
    #Handle the return statement in the target function 
    #Determines the variable values to be returned.
    def visit_Return(self, node):
        return_arrays = []
        if(isinstance(node.value, ast.Tuple)):
            for element in node.value.elts:
                if(isinstance(element, ast.Name)):
                    return_arrays.append(element.id)
            numpy_array_info.add_vars("return",return_arrays)

        elif(isinstance(node.value, ast.Name)):
            numpy_array_info.add_vars("return",[node.value.id])


class _Build_and_lower_mlir:

    declared_index_labels = None
    declared_arrays = None
    id_decl_vars_ta = None
    vals_ta_idx_vars_map = None
    dim_decl_stats_ta = None
    final_id_decl_var = None
    counter = None

    def initialize():
        _Build_and_lower_mlir.declared_index_labels = []
        _Build_and_lower_mlir.declared_arrays = []
        _Build_and_lower_mlir.id_decl_vars_ta = {}
        _Build_and_lower_mlir.vals_ta_idx_vars_map = {}
        _Build_and_lower_mlir.dim_decl_stats_ta = []
        _Build_and_lower_mlir.final_id_decl_var = ""
        _Build_and_lower_mlir.counter = 0

    #Kind of an interface function to build the dimension declarations in the TA dialect
    @classmethod
    def build_dim_decls(self):

        _Build_and_lower_mlir.final_id_decl_var = self.dimensions_to_decl_ta(label_id_map)
        
        return

    #Actual method that builds the dimension declarations in the TA dialect
    @classmethod
    def dimensions_to_decl_ta(self,dims_label_map:dict):

        map_keys = dims_label_map.keys()
        for i,key in enumerate(map_keys):

            if(key not in self.declared_index_labels):

                val = dims_label_map[key]
                if i == 0:
                    lb = "%c{} = arith.constant {} : index".format(i, i)
                    ub = "%c{} = arith.constant {} : index".format(val, val)
                    step = "%cst{} = arith.constant {} : index".format(1, 1)
                    decl = '%{} = "ta.static_index_label"(%c{}, %c{}, %cst{}) : (index, index, index) -> !ta.range'.format(i,i,val,1)
                
                else:
                    lb_const = "%c{}_{}".format(0,i)
                    lb = "{} = arith.constant {} : index".format(lb_const, 0)
                    ub_const = "%c{}_{}".format(i, val)
                    ub = "{} = arith.constant {} : index".format(ub_const, val)
                    step_const = "%cst{}_{}".format(1,i)
                    step = "{}= arith.constant {} : index".format(step_const ,1)
                    decl = '%{} = "ta.static_index_label"({}, {}, {}) : (index, index, index) -> !ta.range'.format(i,lb_const, ub_const,step_const)

                self.id_decl_vars_ta[key] = "%{}".format(i)
                self.vals_ta_idx_vars_map[val] = "%{}".format(i)
                self.declared_index_labels.append(key)
                self.dim_decl_stats_ta.append([lb,ub,step,decl])
            
            if(i == len(map_keys)-1):
                final_var_ta = "%{}".format(i)

        return final_var_ta


     #Method to build the tensor declarations
    @classmethod
    def build_tensors_decls_ta(self):
        count = 0
        tensor_Decl_vars_ta = {}
        list_tensor_decls = []
        last_var_assigned = int(self.final_id_decl_var.split("%")[1]) + 1
    
        #construct tensor declarations for all tensors
        #Determine the tensor type
        for array in list_numpy_arrays:
            if(array not in self.declared_arrays):
                list_array_dims = numpy_array_info.get_dims(array)
                list_array_dims_lbls = numpy_array_info.get_dims_labels(array)
                format = numpy_array_info.get_format(array)
                list_index_vars = []
                tensor_type = "tensor<"
                if list_array_dims != 'No record found':
                    if(list_array_dims_lbls != 'No record found'):  
                        for dimension,dim_lbl in zip(list_array_dims,list_array_dims_lbls):
                            if format == "Dense":
                                tensor_type += "{}x".format(dimension)
                            else:
                                tensor_type += "{}x".format('?')
                                
                            list_index_vars.append(self.id_decl_vars_ta[dim_lbl])
                    else:
                        for dimension in list_array_dims:
                            if format == "Dense":
                                tensor_type += "{}x".format(dimension)
                            else:
                                tensor_type += "{}x".format('?')
                            list_index_vars.append(dimension)

                    tensor_type += "f64>"                                               #Support for float and double data type
                    #Create the variable to be assigned to, i.e. lhs of the decl
                    decl_var = "%{}".format(last_var_assigned+count)
                    #Initialize the declaration builder with the list of indices and tensor type
                    tensor_decl_build_init = PyMLIRGen.Tensor_Decl_Builder(list_index_vars,tensor_type,format)

                    #Build the declaration
                    tensor_decl = decl_var +  tensor_decl_build_init.build_tensor()
                    list_tensor_decls.append(tensor_decl)
                else:
                    tensor_type = "f64"
                    decl_var = "%{}".format(last_var_assigned+count)
                count = count + 1

                tensor_Decl_vars_ta[array] = [decl_var,tensor_type]
                self.declared_arrays.append(array)

        return list_tensor_decls,tensor_Decl_vars_ta


     #Build the tensor fill operations
    def build_tensor_fill_ta(ta_op, value, vartype):
        arr_fill = '"ta.fill"({})'.format(ta_op) + "{value = %s : f64}"%(value) + " : ({})".format(vartype) + " -> ()"
        return arr_fill

    @classmethod
     #Build the labeled tensor and constant operations for random tensor initialization
    def elewise_tensor_fill_ta(self,tensor_to_label,arr_vartype,assigned_val,pred_var_ta, dims):

        lbtensor_ta_var = "%{}".format((int(pred_var_ta.split("%")[1]) + self.counter + 1))   
        const_op_ta_var = "%{}".format((int(pred_var_ta.split("%")[1]) + self.counter + 2))

        elewise_init = PyMLIRGen.ele_wise_fill_Builder(tensor_to_label,lbtensor_ta_var, const_op_ta_var, assigned_val, arr_vartype, dims)
        ele_wise_assign_op = elewise_init.build()

        self.counter += 2
        return const_op_ta_var, ele_wise_assign_op    
    

    def build_op_helper(op_ops, op_types, target_type, dimslbls_lists, target_dims_lbls, id_decl_vars_ta, opr_type, op, op_formats):
        tensor_types = []
        ta_operators = []
        for top in op_ops:
            ta_operators.append(top)
        for type in op_types:
            tensor_types.append(type)


        dimslbls_to_map = []
        for lst in dimslbls_lists:
            dimslbls_to_map = list(set(lst+dimslbls_to_map))

        tc_indices = []
        for dim_lbl in target_dims_lbls:
            tc_indices.append(id_decl_vars_ta[dim_lbl])

        dimslbls_to_map.sort()
        tensor_types.append(target_type)
        builder_init = PyMLIRGen.ArithOp_Builder(ta_operators, dimslbls_to_map, dimslbls_lists, 
                                                                    target_dims_lbls,tensor_types,tc_indices, opr_type,op,op_formats)
        
        return builder_init
        


    #Build the tensor contractions by going through each einsum in order.
    #Can handle dependencies between einsums as well.
    #Other types of tensor operations yet to be handled.
    def build_tensorOps_ta(list_operations:list, pred_var_ta, id_decl_vars_ta:dict, tensor_vars_ta):

        tensor_ops = []
        last_var_assigned = int(pred_var_ta.split("%")[1])+ 1
        count = 0
    #Iterating through each operation in the input, gather the required information 
    #based on the type of operation (either Einsums or tensor arithmetic operations) and build the TA dialect mapping.
        # start_time_op = "%" + str(last_var_assigned+count)
        # start_timer = start_time_op + ' = "ta.getTime"() : () -> f64'
        # tensor_ops.append(start_timer) 
        # count = 1

        tc_var = ""
        for opr_no in list_operations:
            opr_type = oprs_info.get_opr_type(opr_no)
            op = oprs_info.get_op(opr_no)
            target = oprs_info.get_einsum_target(opr_no)
            input_arrays = oprs_info.get_input_arrays(opr_no)
            if(opr_type == "contraction" or opr_type == "transpose" or opr_type == "elewise_mult"):
                input_array_dimslbls_lists = oprs_info.get_input_dims_lbls(opr_no)
                target_dims_lbls = oprs_info.get_out_dims_lbls(opr_no)

                target_var_ta = (tensor_vars_ta[target])[0]
                target_type = (tensor_vars_ta[target])[1]
                
                op1 = input_arrays[0]
                op1_op = tensor_vars_ta[op1][0]
                op1_type = tensor_vars_ta[op1][1]
                op1_dimslbls_lists = input_array_dimslbls_lists[0]
                if len(input_arrays) == 1: # Transpose
                    tc_var = "%{}".format(last_var_assigned+count)
                    format_op1 = numpy_array_info.get_format(op1)
                    format_target = numpy_array_info.get_format(target)
                    tc_builder_init = _Build_and_lower_mlir.build_op_helper([op1_op], [op1_type], target_type, [op1_dimslbls_lists], 
                                                                            target_dims_lbls, id_decl_vars_ta, opr_type, op, [format_op1, format_target])
                    beta_val = PyMLIRGen.ArithOp_Builder.get_beta_val(op)
                    
                    tc_op_ta = tc_var + tc_builder_init.build()

                    set_op = '\n"ta.set_op"({},{}) '.format(tc_var,target_var_ta) \
                                    + "{" + "__beta__ = {} : f64".format(beta_val) +'} : ' +'({},{})'.format(target_type,target_type) + ' -> ()'
                    tensor_ops.append(tc_op_ta + set_op)

                    count += 1
                else:
                    format_target = numpy_array_info.get_format(op1)
                    for jj, op2 in enumerate(input_arrays[1:]):
                        j = jj + 1
                        tc_var = "%{}".format(last_var_assigned+count)

                        op2_dimslbls_lists = input_array_dimslbls_lists[j]
                        int_input_array_dimslbls_lists = [op1_dimslbls_lists, op2_dimslbls_lists]
                        if j == len(input_arrays) - 1:
                            int_target_dims = target_dims_lbls
                            int_target_type = target_type
                        else:
                            int_target_dims = [x for x in op1_dimslbls_lists if x not in op2_dimslbls_lists]
                            int_target_dims += [x for x in op2_dimslbls_lists if x not in op1_dimslbls_lists]
                            int_target_type = "tensor<"+ "x".join(["{}"] * len(int_target_dims)).format(*[label_id_map[x] for x in int_target_dims])+"xf64>"

                        format_op1 = format_target
                        format_op2 = numpy_array_info.get_format(op2)
                        if opr_type == "contraction":
                            format_target = sp_matmult_conversions[format_target][format_op2]
                        elif opr_type == "elewise_mult":
                            format_target = sp_elw_mult_conversions[format_target][format_op2]
                        tc_builder_init = _Build_and_lower_mlir.build_op_helper([op1_op,tensor_vars_ta[op2][0]], [op1_type, tensor_vars_ta[op2][1]], int_target_type, int_input_array_dimslbls_lists, 
                                                                            int_target_dims, id_decl_vars_ta, opr_type, op, [format_op1, format_op2, format_target])
                        
                        tc_op_ta = tc_var + tc_builder_init.build()

                        op1_dimslbls_lists =  int_target_dims
                        op1_op = tc_var

                        tensor_ops.append(tc_op_ta)
                        count += 1

                    beta_val = PyMLIRGen.ArithOp_Builder.get_beta_val(op)
                    set_op = '"ta.set_op"({},{}) '.format(tc_var,target_var_ta) \
                                    + "{" + "__beta__ = {} : f64".format(beta_val) +'} : ' +'({},{})'.format(target_type,target_type) + ' -> ()'
                    tensor_ops.append(set_op)
         
            elif(opr_type == "tensor_sum"):
                target_var_ta = (tensor_vars_ta[target])[0]
                target_type = (tensor_vars_ta[target])[1]
                op1 = input_arrays[0]
                op1_op = tensor_vars_ta[op1][0]
                op1_type = tensor_vars_ta[op1][1]
                sum_builder = PyMLIRGen.TensorSumBuilder([op1_op], [op1_type], [target_type])
                tensor_ops.append(target_var_ta + sum_builder.build())

            #Handle other tensor operations like tensor arithmetic operations
            else:
                ta_operators = []
                tensor_types = []
                tc_var = "%{}".format(last_var_assigned+count)

                target_var_ta = (tensor_vars_ta[target])[0]
                target_type =  (tensor_vars_ta[target])[1]

                for array in input_arrays:
                    ta_operators.append((tensor_vars_ta[array])[0])
                    tensor_types.append((tensor_vars_ta[array])[1])

                input_array_dimslbls_lists = oprs_info.get_input_dims_lbls(opr_no)
                target_dims_lbls = oprs_info.get_out_dims_lbls(opr_no)
                
                ta_arith_build_init = _Build_and_lower_mlir.build_op_helper(ta_operators, tensor_types, target_type, input_array_dimslbls_lists, target_dims_lbls, id_decl_vars_ta, opr_type, op, [numpy_array_info.get_format(input_arrays[0]), numpy_array_info.get_format(input_arrays[1]), numpy_array_info.get_format(input_arrays[0])])
                arith_op_ta = tc_var + ta_arith_build_init.build()
                beta_val =  0.000000e+00
                set_op = '\n"ta.set_op"({},{})'.format(tc_var,target_var_ta) + \
                                "{" + "__beta__ = {} : f64".format(beta_val) +'} : ' +'({},{})'.format(target_type,target_type) + ' -> ()'

                tensor_ops.append(arith_op_ta + set_op)
        
                count += 1

        # end_time_op = "%" + str(last_var_assigned+count)
        # end_timer = end_time_op + ' = "ta.getTime"() : () -> f64'
        # print_time = '"ta.print_elapsed_time"({}, {})'.format(start_time_op,end_time_op)  + ' : (f64, f64) -> ()'
        # tensor_ops.append(end_timer)
        # tensor_ops.append(print_time)
        return tensor_ops


    #Build the tensor contractions in TA dialect and lower
    @classmethod
    def build_tensorOps_and_lower(self,irb,tensor_vars_ta, final_tensor_decl_var,flags, with_jit, func_name, args_vals, outputs):

        #Proceed to creating tensor fill operations or ops for random initialization
        final_var_ta = ""
        for label in tensor_vars_ta.keys():
            # if label in numpy_array_info.get_vars("args"):
            #     continue
            ta_op_type_list = tensor_vars_ta[label]
            ta_op_var = ta_op_type_list[0]
            vartype = ta_op_type_list[1]
            init_value = numpy_array_info.get_val(label)
            if(scp.sparse.issparse(init_value)):
                irb.add_statement('"ta.fill_from_file"({}) {{filename = "SPARSE_FILE_NAME0", readMode = 1 : i32}} : ({}) -> ()'.format(ta_op_var ,vartype))               
            else:    
                if(isinstance(init_value, np.ndarray)):
                    temp_arr = init_value.reshape(1,init_value.size)
                    init_value = temp_arr[0][0]
                    ta_fillOp = _Build_and_lower_mlir.build_tensor_fill_ta(ta_op_var,init_value, vartype)
                    irb.add_statement(ta_fillOp)

                elif(isinstance(init_value, float)):
                    ta_fillOp = _Build_and_lower_mlir.build_tensor_fill_ta(ta_op_var,init_value,vartype)
                    irb.add_statement(ta_fillOp)

                elif(init_value == None):
                    pass
                else:
                    raise RuntimeError(
                        "Unsupported input type"
                    )
                
        #Proceed to generate the Tensor contraction operations
        if(final_var_ta):
            tensor_ops_ta = _Build_and_lower_mlir.build_tensorOps_ta(list_operations, final_var_ta, self.id_decl_vars_ta, tensor_vars_ta)
        else:
            tensor_ops_ta = _Build_and_lower_mlir.build_tensorOps_ta(list_operations, final_tensor_decl_var,self.id_decl_vars_ta, tensor_vars_ta)            

        for tc_op_ta in tensor_ops_ta:
            irb.add_statement(tc_op_ta)
    

        #Add print statements for testing purposes in accordance with the input. 
        list_out_dims = []
        if numpy_array_info.get_vars("return") != "No record found":
            for target in numpy_array_info.get_vars("return"):
                out_tensor = (tensor_vars_ta[target])[0]
                outputtype = (tensor_vars_ta[target])[1]
                list_out_dims.append(numpy_array_info.get_dims(target))
                irb.add_statement('"ta.print"({}) : ({}) -> ()'.format(out_tensor, outputtype) )
                # irb.add_statement('return {} : '.format(out_tensor) + '{}'.format(outputtype))
        
        # irb.add_statement('"ta.return"() : () -> ()')
        irb.add_statement('return')
        #Return the generated dialect

        ta_dialect = PyMLIRGen.MLIRFunctionBuilder.compile(irb)

        # If multiplying with sparse this flag is required
        if need_opt_comp_workspace:
            if flags:
                flags += '--opt-comp-workspace'
            else:
                flags = '--opt-comp-workspace'
        
        #Begin lowering. TODO: Correctness check for enabled flags.
        if with_jit:

            lowering_result = lowering.lower_dialect_with_jit(ta_dialect,list_out_dims,flags,func_name, args_vals, outputs)
        else:
            lowering_result = lowering.lower_dialect(ta_dialect,list_out_dims,flags,func_name)
      
        return lowering_result
    
       

#Wrapper function. The input function (in the form of an object) is passed an arguement to this function.
def compile(flags, with_jit=True):
    
    def innerfunc(func):

        def wrapper(*pos_args, **kwargs):
            global sp_matmult_conversions
            global sp_mattr_conversions
            global sp_elw_mult_conversions
            global sp_elw_add_sub_conversions
            
            # Output formats when multiply matrices of different formats
            sp_matmult_conversions = {
                "CSR": {
                    "CSR" : "CSR",
                    "COO" : "CSR",
                    "Dense" : "Dense",
                },
                "COO": {
                    "CSR" : "CSR",
                    "COO" : "CSR",
                    "Dense" : "Dense"
                },
                "Dense": {
                    "CSR" : "Dense",
                    "COO" : "Dense",
                    "Dense" : "Dense",
                }
            }

            # Output formats when transposing matrices of different formats
            sp_mattr_conversions = {
                "CSR" : "CSC",
                "COO" : "COO",
                "Dense" : "Dense"
            }

            # Output formats when elwise mult matrices of different formats
            sp_elw_mult_conversions = {
                "CSR": {
                    "CSR" : "CSR",
                    "COO" : "CSR",
                    "Dense" : "COO",
                },
                "COO": {
                    "CSR" : "CSR",
                    "COO" : "CSR",
                    "Dense" : "COO"
                },
                "Dense": {
                    "CSR" : "Dense",
                    "COO" : "Dense",
                    "Dense" : "Dense",
                }
            }
            # Output formats when elwise add or subtract matrices of different formats
            # Add and subtract is almost the same as elwise mult with two differences
            sp_elw_add_sub_conversions = sp_elw_mult_conversions


            sp_elw_add_sub_conversions["CSR"]["Dense"] = "Dense"
            sp_elw_add_sub_conversions["COO"]["Dense"] = "Dense"


            func_str = inspect.getsource(func)
            parsed_func = ast.parse(func_str)
            v = _AnalysisNodeVisitor()

            global label_id_map
            label_id_map = {}
            global list_operations
            list_operations = []
            global list_numpy_arrays
            list_numpy_arrays = []
            global numpy_array_info
            numpy_array_info = HashTable(50)
            global args_vals
            args_vals = []
            global need_opt_comp_workspace
            need_opt_comp_workspace = False

            #Parse the function body, create the AST and generate the TA dialect using the AST
            if(isinstance(parsed_func.body[0], ast.FunctionDef)):
                func_def = parsed_func.body[0]
                #If the numpy arrays are passed as arguements to the function
                in_args = []
                for func_arg,value in zip(func_def.args.args, pos_args):
                    numpy_array_label = func_arg.arg
                    if scp.sparse.issparse(value):
                        if scp.sparse.isspmatrix_csr(value):
                            numpy_array_info.set_format(numpy_array_label, "CSR")
                        elif scp.sparse.isspmatrix_coo(value):
                            numpy_array_info.set_format(numpy_array_label, "COO")
                        elif scp.sparse.isspmatrix_csc(value):
                            numpy_array_info.set_format(numpy_array_label, "CSC")
                        else:
                            raise RuntimeError(numpy_array_label+
                                " Unsupported sparsity format"
                            )
                    else:
                        numpy_array_info.set_format(numpy_array_label, "Dense")
                    numpy_array_info.set_val(numpy_array_label, value)
                    numpy_array_info.set_dims(numpy_array_label, list(value.shape))
                    in_args.append(numpy_array_label)
                    list_numpy_arrays.append(numpy_array_label)
                    args_vals.append(value)

                numpy_array_info.add_vars("args", in_args)
                
                #Initialize the global variables of the class
                _Build_and_lower_mlir.initialize()
                #Visit each node and gather the required information about the tensor computation, the input arrays and the targets
                #At the same time, build the index label declarations
                
                v.visit(parsed_func)

                #Build the dimension declarations for the dialect
                _Build_and_lower_mlir.build_dim_decls()
                

                # irb = PyMLIRGen.MLIRFunctionBuilder(
                #     func_def.name,
                #     return_types=[],
                # )    

                #Add the dimension declaration statements to the TA dialect function 
                # for st_list in _Build_and_lower_mlir.dim_decl_stats_ta:
                #     for st in st_list:
                #         irb.add_statement(st)

                all_st_list = _Build_and_lower_mlir.dim_decl_stats_ta

                #Build the tensor declarations in the TA dialect
                list_tensor_decls_ta, tensor_vars_ta = _Build_and_lower_mlir.build_tensors_decls_ta()
                
                final_tensor_decl_var = (list(tensor_vars_ta.values())[-1])[0]
                
                list_out_dims = []

                # Retrieve the functions return types to populate the function signature
                # Currently, COMET does not like this, so it's not used.
                # for target in numpy_array_info.get_vars("return"):
                #     outputtype = (tensor_vars_ta[target])[1]
                #     print("Got return type {}".format(outputtype) )
                #     list_out_dims.append(outputtype)
                list_in_dims = []
                for target in numpy_array_info.get_vars("args"):
                    arg = (tensor_vars_ta[target])[0]
                    outputtype = (tensor_vars_ta[target])[1]
                    list_in_dims.append((arg, outputtype))
                
                outputs = []
                if numpy_array_info.get_vars("return") != "No record found":
                    for target in numpy_array_info.get_vars("return"):
                        if target in tensor_vars_ta: 
                            arg = (tensor_vars_ta[target])[0]
                            format = numpy_array_info.get_format(target)
                            if format == "Dense":
                                outputtype = (tensor_vars_ta[target])[1]
                                if 'x' in outputtype:
                                    shape = [int(x) for x in outputtype.split('<')[1].split('f64>')[0].split('x')[:-1]]
                                else:
                                    shape = [1,]
                                list_in_dims.append((arg, outputtype))
                                outputs.append(np.empty(shape))
                            else:
                                if format == "CSR":
                                    outputs.append(scp.sparse.csr_matrix([]))
                                elif format == "CSC":
                                    outputs.append(scp.sparse.csc_matrix([]))
                                elif format == "COO":
                                    outputs.append(scp.sparse.coo_matrix([]))
                        # else: # If we only return a value e.g. after a sum


                irb = PyMLIRGen.MLIRFunctionBuilder(
                    func_def.name,
                    input_types=list_in_dims,
                    return_types=list_out_dims,
                ) 

                #Add the dimension declaration statements to the TA dialect function 
                for st_list in all_st_list:
                    for st in st_list:
                            irb.add_statement(st)

                #Add the tensor declaration statements to the TA dialect function 
                for decl in list_tensor_decls_ta:
                    irb.add_statement(decl)

                #Build the tensor operations and lower
                result = _Build_and_lower_mlir.build_tensorOps_and_lower(irb,tensor_vars_ta,final_tensor_decl_var,flags, with_jit, func_def.name, args_vals, outputs)
            
            return result
        
        return wrapper
    return innerfunc
