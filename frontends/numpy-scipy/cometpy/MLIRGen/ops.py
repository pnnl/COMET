import jinja2

from cometpy.MLIRGen.types import *
from cometpy.MLIRGen.utils import *
import scipy as sp


class Dialect:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name} dialect"


class Operation:
    
    def __init__(self, block, index_in_block, operands, result_types, startsBody = False, endsBody = False):
        self.startsBody = startsBody
        self.endsBody = endsBody
        self.operands = operands
        self.index_in_block = index_in_block
        self.block = block
        self.results = [Symbol(t, self, block) for t in result_types]


class BinaryElementwiseOp(Operation):

    def __init__(self, block, iiblock, lhs, rhs, _type = None):
        super().__init__(block, iiblock, [lhs, rhs], [_type if _type else lhs.type])
        self.lhs = lhs
        self.rhs = rhs
    
    def dump(self, name):
        return self.binary_op_text.render(
            ssa = self.results[0].ssa,
            name = name,
            lhs = self.lhs.ssa,
            rhs = self.rhs.ssa,
            type = self.results[0].type
        )


    binary_op_text = jinja2.Template(
        '%{{ssa}} = {{name}} %{{lhs}}, %{{rhs}}: {{type}}',
        undefined=jinja2.StrictUndefined,
    )

class SubOp(BinaryElementwiseOp):

    def __init__(self, block, iiblock, lhs, rhs):
        super().__init__(block, iiblock, lhs, rhs)

    def dump(self):
        name = self.text.render(
            type_suffix = 'i' if self.lhs.type == 'index' else 'f',
        )
        return super().dump(name)

    text = jinja2.Template(
        'arith.sub{{type_suffix}}',
        undefined=jinja2.StrictUndefined,
    )

class MulOp(BinaryElementwiseOp):

    def __init__(self, block, iiblock, lhs, rhs):
        super().__init__(block, iiblock, lhs, rhs)

    def dump(self):
        name = self.text.render(
            type_suffix = 'i' if self.lhs.type == 'index' else 'f',
        )
        return super().dump(name)

    text = jinja2.Template(
        'arith.mul{{type_suffix}}',
        undefined=jinja2.StrictUndefined,
    )

class TensorIndexBasedOp(Operation):
    def __init__(self, block, iiblock, name, inputs, inputs_indices, res_indices, res_format, res_val = None, alpha = None, beta = None , mask = None, mask_type = None, semiring = None):
        self.name = name
        self.inputs = inputs
        self.inputs_indices = inputs_indices
        self.mask_type = mask_type
        self.semiring = semiring
        self.alpha = alpha
        self.beta = beta
        self.mask = mask
        self.res_val = res_val

        self.res_indices = res_indices
        all_indices = inputs_indices[0][:]
        for indices in inputs_indices:
            for index in indices:
                if not index in all_indices:
                    all_indices.append(index)

        indices_to_dims = {}
        for i, index in enumerate(all_indices):
            indices_to_dims[index] = i
        
        res_shape = []
        for res_index in self.res_indices:
            found = False
            for input_id, indices in enumerate(inputs_indices):
                for dim, index in enumerate(indices):
                    if res_index == index:
                        res_shape.append(inputs[input_id].type.shape[dim])
                        found = True
                        break
                if found:
                    break
            
            if found:
                    continue
            
        affine_map_all = '{}'.format(",".join([f'd{d}' for d in range(len(all_indices))]))
            
        affine_map_res = 'affine_map<({}) -> ({})>'.format(affine_map_all, ",".join([f'd{indices_to_dims[index]}' for index in self.res_indices]))
        affine_maps = []
        for indices in inputs_indices:
            affine_map_input = 'affine_map<({}) -> ({})>'.format(affine_map_all,  ",".join([f'd{indices_to_dims[index]}' for index in indices]))
            affine_maps.append(affine_map_input)

        affine_maps.append(affine_map_res)
        self.indexing_maps = ",".join(affine_maps)
        if res_format != DENSE:
            for input in inputs:
                if isinstance(input.type, TASparseTensorType):
                    indices_type = input.type.indices_type
            res_type = TASparseTensorType(res_shape, inputs[0].type.element_type, indices_type, res_format)
        else:
            res_type = TensorType(res_shape, inputs[0].type.element_type)
        if self.res_val:
            super().__init__(block, iiblock, [*inputs, *inputs_indices, res_indices, self.res_val], [res_type])
        else:
            super().__init__(block, iiblock, [*inputs, *inputs_indices, res_indices], [res_type])

    def dump(self):
        input_index_label_nums = ", ".join([str(len(indices)) for indices in self.inputs_indices])
        all_index_label_nums = ", ".join([input_index_label_nums, (str(len(self.res_indices)))])
        input_types = ", ".join([
                ",".join([f'{input.type}' for input in self.inputs])
                , ", ".join([f'{index.type}' for indices in self.inputs_indices for index in indices])
                , ", ".join([f'{oper.type}' for oper in self.res_indices])]) 
        if self.mask != None:
            input_types += f", {self.mask.type}"
        if self.res_val != None:
            input_types += f", {self.res_val.type}"
        if self.mask_type:
            segment_sizes = ", ".join([", ".join('1' * len(self.inputs)), all_index_label_nums, '1' if self.res_val else '0', '1' if self.mask_type != 'none' else '0'])
        else:
            segment_sizes = ", ".join([", ".join('1' * len(self.inputs)), all_index_label_nums, '1' if self.res_val else '0'])
        return self.tensor_binary_op_text.render(
            
            ssa = self.results[0].ssa,
            name = self.name,
            inputs = ", ".join([f'%{input.ssa}' for input in self.inputs]),
            inputs_indices = ", ".join([f'%{index.ssa}' for indices in self.inputs_indices for index in indices]),
            res_indices = ", ".join([f'%{input.ssa}' for input in self.res_indices]),
            mask_type = self.mask_type,
            mask = self.mask,
            semiring = self.semiring,
            input_types = input_types,
            res_type = self.results[0].type,
            affine_maps = self.indexing_maps,
            operand_segment_sizes = segment_sizes,
            res_val = self.res_val,
            alpha = self.alpha,
            beta = self.beta,
        )
    
    tensor_binary_op_text = jinja2.Template(
        '%{{ssa}} = "{{name}}" ({{inputs}}, {{inputs_indices}}, {{res_indices}}{%if res_val!=None%}, %{{res_val.ssa}}{%endif%}{%if mask!=None%}, %{{mask.ssa}}{%endif%}) <{ {% if mask_type%} MaskType = "{{mask_type}}", {% endif%} indexing_maps = [{{affine_maps}}], operandSegmentSizes = array<i32: {{operand_segment_sizes}}>{% if semiring%} , semiring="{{semiring}}" {%endif%}}>  {%if alpha != None   or beta != None %} { {%endif%} {%if alpha!= None%} __alpha__ = {{alpha}} : f64 {%endif%} {%if alpha!= None and beta!= None %}, {%endif%} {%if beta!= None %} __beta__ = {{beta}}: f64 {%endif%} {%if alpha!= None or beta!= None %}} {%endif%} : ({{input_types}}) -> {{res_type}}',
        undefined=jinja2.StrictUndefined,
    )

class TensorBinaryOp(TensorIndexBasedOp):

    def __init__(self, block, iiblock, name, lhs, rhs, lhs_indices, rhs_indices, res_indices, res_val, alpha, beta, mask, masktype, semiring, res_format):
        super().__init__(block, iiblock, name, [lhs, rhs], [lhs_indices, rhs_indices], res_indices, res_format, res_val, alpha, beta, mask, masktype, semiring)
        self.lhs = lhs
        self.lhs_indices = lhs_indices
        self.rhs = rhs
        self.rhs_indices = rhs_indices
        self.res_indices = res_indices
        self.mask_type = masktype
        self.semiring = semiring


    def dump(self):
        return super().dump()

class TensorElewiseBinaryOp(TensorBinaryOp):

    def __init__(self, block, iiblock, name, lhs, rhs, lhs_indices, res_val, alpha, beta, semiring, format):
        super().__init__(block, iiblock, name, lhs, rhs, lhs_indices, lhs_indices, lhs_indices, res_val, alpha, beta, None, None, semiring, format)

    def dump(self):
        return super().dump()


class TensorIndexLabelOp(Operation):

    def __init__(self, block, iiblock):
        super().__init__(block, iiblock, [], [TensorIndexType()])

    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            res_type = self.results[0].type
        )

    text = jinja2.Template (
        '%{{ssa}} = "ta.index_label" () : () -> {{res_type}}'
    )

class TensorAddOp(TensorElewiseBinaryOp):

    def __init__(self, block, iiblock, lhs, rhs, res_indices, format, res_val, alpha = 1.0, beta= 0.0, semiring= "noop_plusxy"):
        super().__init__(block, iiblock, 'ta.add', lhs, rhs, res_indices, res_val, alpha, beta, semiring, format)

    def dump(self):
        return super().dump()

class TensorSubOp(TensorElewiseBinaryOp):

    def __init__(self, block, iiblock, lhs, rhs, res_indices, format, res_val, alpha = 1.0, beta= 0.0, semiring= "noop_minus"):
        super().__init__(block, iiblock, 'ta.subtract', lhs, rhs, res_indices, res_val, alpha, beta, semiring, format)

    def dump(self):
        return super().dump()

class TensorMulOp(TensorElewiseBinaryOp):

    def __init__(self, block, iiblock, lhs, rhs, res_indices, format, res_val, alpha = 1.0, beta= 0.0, semiring= "noop_times"):
        super().__init__(block, iiblock, 'ta.elews_mul', lhs, rhs, res_indices, res_val, alpha, beta, semiring, format)

    def dump(self):
        return super().dump()

class TensorTransposeOp(TensorIndexBasedOp):

    def __init__(self, block, iiblock, input, input_indices, res_indices, res_format, res_val):
        super().__init__(block, iiblock, 'ta.transpose', [input], [input_indices], res_indices, res_format, res_val)
        self.input = input
        self.input_indices = input_indices
        self.res_indices = res_indices

    def dump(self):
        return super().dump()


class TensorSumOp(Operation):

    def __init__(self, block, iiblock, value):
        super().__init__(block, iiblock, [value], [value.type.element_type])
        self.value = value

    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            value = self.value.ssa,
            val_type = self.value.type,
            res_type = self.results[0].type
        )

    text = jinja2.Template(
        '%{{ssa}} = "ta.reduce"(%{{value}}) : ({{val_type}}) -> {{res_type}}'
    )

class TensorSetOp(Operation): 

    def __init__(self, block, iiblock, src, dst, beta):
        super().__init__(block, iiblock, [src, dst], [])
        self.src = src
        self.dst = dst
        self.beta = beta
    
    def dump(self):
        return self.text.render (
            src = self.src.ssa,
            dst = self.dst.ssa,
            beta = self.beta,
            beta_type = 'f64',
            src_type = self.src.type,
            dst_type = self.dst.type,
        )


    text = jinja2.Template(
        '"ta.set_op" (%{{src}}, %{{dst}}) {__beta__ = {{beta}} : {{beta_type}} } : ({{src_type}}, {{dst_type}}) -> ()'
    )

class TensorPrintOp(Operation):

    def __init__(self, block, iiblock, input):
        super().__init__(block, iiblock, [input], [])
        self.input = input
    
    def dump(self):
        return self.text.render (
            input = self.input.ssa,
            input_type = self.input.type
        )


    text = jinja2.Template(
        '"ta.print" (%{{input}}) : ({{input_type}}) -> ()'
    )


class TensorMatMultOp(TensorBinaryOp):

    def __init__(self, block, iiblock, lhs, rhs, lhs_indices, rhs_indices, res_indices, res_format, res_val, alpha = 1.0, beta = 0.0, mask = None, masktype='none', semiring='plusxy_times'):
        super().__init__(block, iiblock, 'ta.mul', lhs, rhs, lhs_indices, rhs_indices, res_indices, res_val, alpha, beta, mask, masktype, semiring, res_format)


class AddOp(BinaryElementwiseOp):

    def __init__(self, bloc, iiblock, lhs, rhs):
        super().__init__(bloc, iiblock, lhs, rhs)

    def dump(self):
        name = self.text.render(
            type_suffix = 'i' if self.lhs.type == 'index' else 'f',
        )
        return super().dump(name)

    text = jinja2.Template(
        'arith.add{{type_suffix}}',
        undefined=jinja2.StrictUndefined,
    )
    

class ConstantOp(Operation):

    def __init__(self, block, iiblock, val):
        self.value = val

        if isinstance(val, int):
            _type = 'index'
        elif isinstance(val, float):
            _type = 'f64'
        
        super().__init__(block, iiblock, [], [_type])

    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            value = self.value,
            type = self.results[0].type
        )

    text = jinja2.Template(
        '%{{ssa}} = arith.constant {{value}}: {{type}}',
        undefined=jinja2.StrictUndefined,
    )

class ToMemrefOp(Operation):

    def __init__(self, block, iiblock, src):
        super().__init__(block, iiblock, [src], [MemrefType(src.type.shape, src.type.element_type)])
        self.src = src


    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            src = self.src.ssa,
            res_type = self.results[0].type,
        )


    text = jinja2.Template (
        '%{{ssa}} = bufferization.to_memref %{{src}}: {{res_type}} '
    )

class ToTensorOp(Operation):

    def __init__(self, block, iiblock, src):
        super().__init__(block, iiblock, [src], [TensorType(src.type.shape, src.type.element_type)])
        self.src = src


    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            src = self.src.ssa,
            src_type = self.src.type,
        )


    text = jinja2.Template (
        '%{{ssa}} = bufferization.to_tensor %{{src}} restrict writable : {{src_type}} '
    )


class LoadOp(Operation):

    def __init__(self, block, iiblock, src, indices):
        super().__init__(block, iiblock, [src] + indices, [src.type.element_type])
        self.src = src
        self.indices = indices

    def dump(self):
        return self.text.render(
            ssa = self.results[0].ssa,
            src = self.src.ssa,
            indices = ",".join(["%{}".format(index.ssa) for index in self.indices]),
            src_type = self.src.type
        )

    text = jinja2.Template(
        '%{{ssa}} = memref.load %{{src}}[{{indices}}]: {{src_type}}',
        undefined=jinja2.StrictUndefined,
    )

class StoreOp(Operation):

    def __init__(self, block, iiblock, value, dst, indices):
        super().__init__(block, iiblock, [value, dst] + indices, [])
        self.value = value
        self.dst = dst
        self.indices = indices

    def dump(self):
        return self.text.render(
            value = self.value.ssa,
            dst = self.dst.ssa,
            indices = ",".join(["%{}".format(index.ssa) for index in self.indices]),
            dst_type = self.dst.type
        )

    text = jinja2.Template(
        'memref.store  %{{value}}, %{{dst}}[{{indices}}]: {{dst_type}}',
        undefined=jinja2.StrictUndefined,
    )


class Symbol:
    curr_ssa = 0

    def __init__(self, _type, defining_op, block):
        self.ssa = Symbol.curr_ssa
        self.type = _type
        self.defining_op = defining_op
        self.block = block
        Symbol.curr_ssa += 1

class SymbolTable:

    def __init__(self):
        self.table = {}
        Symbol.curr_ssa = 0

    def insert(self, id, symbol):
        self.table[id] = symbol

    def find(self, id):
        if id in self.table:
            return self.table[id]
        else:
            return None

class YieldOp(Operation):

    def __init__(self, block, iiblock, values):
        super().__init__(block, iiblock, values, [v.type for v in values], endsBody=True)
        self. values = values

    def dump(self):
        return self.text.render(
            values = ",".join([f'%{v}' for v in self.values]),
        )


    text = jinja2.Template(
        'scf.yield {% if values %} ({{values}}) -> () {% endif%}',
        undefined=jinja2.StrictUndefined,
    )
    
class ReduceOp(Operation):

    def __init__(self, block, iiblock, values):
        super().__init__(block, iiblock, values, [v.type for v in values], endsBody=True)
        self. values = values

    def dump(self):
        return self.text.render(
            values = ",".join([f'%{v}' for v in self.values]),
        )


    text = jinja2.Template( 
        'scf.reduce {% if values %} ({{values}}) -> () {% endif%}',
        undefined=jinja2.StrictUndefined,
    )
    

class ReturnOp(Operation):

    def __init__(self, block, iiblock, values):
        super().__init__(block, iiblock, values, [v.type for v in values], endsBody=True)
        self. values = values

    def dump(self):
        return self.text.render(
            values = ",".join([f'%{v.ssa}' for v in self.values]),
            ret_types = ",".join([f'{v.type}' for v in self.values]),
        )


    text = jinja2.Template(
        'func.return {% if values %} {{values}} : {{ret_types}} {% endif%}',
        undefined=jinja2.StrictUndefined,
    )

class OperationWithBody(Operation):

    def __init__(self, block, iiblock, operands, return_types):
        super().__init__(block, iiblock, operands, return_types, True)
        self.body = []
        self.identation  = 0

    def dump(self) -> str:
        statements = "\n".join([" "*self.identation + stmt.dump() for stmt in self.body])
        return self.body_text.render(
            identation = " "* (self.identation - 2),
            statements = statements
        )
    
    body_text = jinja2.Template(
        " {"
        + "\n"
        + "{{statements}}"
        + "\n"
        + "{{identation}}"
        + "}",
        undefined=jinja2.StrictUndefined,
    )
    

class ModuleOp(OperationWithBody):

    def __init__(self, block, iiblock):
        super().__init__(block, iiblock, [], [])

    def dump(self) -> str:
        op = self.text.render()
        statements = super().dump() 
        return op + statements
    
    text = jinja2.Template(
        "module",
        undefined=jinja2.StrictUndefined,
    )

class FuncOp(OperationWithBody):

    function_text = jinja2.Template(
        "func.func {% if private %} private {% endif %}@{{func_name}}({{inputs}})  -> ({{return_types}}) attributes {llvm.emit_c_interface}",
        undefined=jinja2.StrictUndefined,
    )

    def __init__(
        self,
        block,
        iiblock,
        func_name: str,
        inputs,
        return_types,
        private: bool,
    ) -> None:
        super().__init__(block, iiblock, [], return_types)
        self.func_name = func_name
        self.inputs = inputs
        self.return_types = return_types
        self.private = private
    

    def dump(self) -> str:
        return self.function_text.render(
            func_name = self.func_name,
            inputs = ", ".join(["%{}: {}".format(input.ssa, input.type) for input in self.inputs]),
            return_types = ", ".join(["{}".format(ret) for ret in self.return_types]),
            private = self.private
        ) + super().dump()


class ForOp(OperationWithBody):

    def __init__(self, block, iiblock, lb, ub, step):
        super().__init__(block, iiblock, [lb, ub, step], [])
        self.lb = lb
        self.ub = ub
        self.step = step
        self.iv = Symbol('index', None, self)

    def dump(self):
        op =  self.text.render(
            iv = self.iv.ssa,
            lb = self.lb.ssa,
            ub = self.ub.ssa,
            step = self.step.ssa,
        )
        return op + super().dump()


    text = jinja2.Template(
        'scf.for %{{iv}} = %{{lb}} to %{{ub}} step %{{step}}',
        undefined=jinja2.StrictUndefined,
    ) 


class ForAllOp(OperationWithBody):

    def __init__(self, block, iiblock, lb, ub, step):
        super().__init__(block, iiblock, [lb, ub, step], [])
        self.lb = lb
        self.ub = ub
        self.step = step
        self.iv = Symbol('index', None, self)

    def dump(self):
        my_render = self.text.render(
            iv = self.iv.ssa,
            lb = self.lb.ssa,
            ub = self.ub.ssa,
            step = self.step.ssa,
        ) 
        body_render = super().dump()
        return my_render + body_render


    text = jinja2.Template(
        'scf.parallel (%{{iv}}) = (%{{lb}}) to (%{{ub}}) step (%{{step}}) ',
        undefined=jinja2.StrictUndefined,
    )
