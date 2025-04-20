from  cometpy.MLIRGen.lowering import types
import scipy as sp
import ctypes
from ctypes import *
def create_memref_type(type, rank):
    class memref_template(Structure):
        _fields_ = [ ('mem_aligned', POINTER(type)), ('mem', POINTER(type)), ('offset', c_int64), ('dims', c_int64 * rank), ('stride', c_int64 * rank)]
    
    return memref_template

class memref_i64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_int64)), ('mem', POINTER(c_int64)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_i32(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_int32)), ('mem', POINTER(c_int32)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_f64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_double)), ('mem', POINTER(c_double)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_f32(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_float)), ('mem', POINTER(c_float)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]


# llvm_args += [*expand_memref_ptr(dim_sizes), 0, *expand_memref_ptr(A1pos), 0, *expand_memref_ptr(A2pos), *expand_memref_ptr(A2crd), *expand_memref_ptr(Aval)]

class output_csr_f64_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('insert_2', c_int64), ('A2pos', memref_i64), ('A2crd', memref_i64), ('Aval', memref_f64)]
    
class output_csr_f32_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('insert_2', c_int64), ('A2pos', memref_i64), ('A2crd', memref_i64), ('Aval', memref_f32)]

class output_coo_f64_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('A1crd', memref_i64), ('insert_2', c_int64), ('A2crd', memref_i64), ('Aval', memref_f64)]

class output_coo_f32_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('A1crd', memref_i64), ('insert_2', c_int64), ('A2crd', memref_i64), ('Aval', memref_f32)]

class output_csr_f64_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('insert_2', c_int64), ('A2pos', memref_i32), ('A2crd', memref_i32), ('Aval', memref_f64)]
    
class output_csr_f32_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('insert_2', c_int64), ('A2pos', memref_i32), ('A2crd', memref_i32), ('Aval', memref_f32)]

class output_coo_f64_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('A1crd', memref_i32), ('insert_2', c_int64), ('A2crd', memref_i32), ('Aval', memref_f64)]

class output_coo_f32_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('A1crd', memref_i32), ('insert_2', c_int64), ('A2crd', memref_i32), ('Aval', memref_f32)]



def get_tensor_type(datatype, shape, format, indices_type):
    if format != types.DENSE:
        tensor_formats = []
        if format == types.CSR:
            tensor_formats.append("d")
            tensor_formats.append("unk")
            tensor_formats.append("cu")
            tensor_formats.append("unk")
        elif format == types.COO:
            tensor_formats.append("cn")
            tensor_formats.append("unk")
            tensor_formats.append("s")
            tensor_formats.append("unk")
        return "!ta.sparse_tensor<{}, {}, {}, {}>".format(datatype, indices_type,"x".join(str(v) for v in shape), ",".join(f for f in tensor_formats))
    else:
        return "tensor<{}x{}>".format("x".join(str(v) for v in shape), datatype)



def memref_type_from_dense_ndarray(ndarray):
    return types.MemrefType(ndarray.shape, dtype_to_mlir_type(ndarray.dtype))

def tensor_type_from_dense_ndarray(ndarray):
    return types.TensorType(ndarray.shape, dtype_to_mlir_type(ndarray.dtype))

def mlir_type_from_sparse_ndarray(sp_ndarray):
    format = None
    if sp_ndarray.format == 'csr':
        format = types.CSR
        return types.TASparseTensorType(sp_ndarray.shape, dtype_to_mlir_type(sp_ndarray.dtype), dtype_to_mlir_type(sp_ndarray.indices.dtype), format)
    elif sp_ndarray.format == 'coo':
        format = types.COO
        return types.TASparseTensorType(sp_ndarray.shape, dtype_to_mlir_type(sp_ndarray.dtype), dtype_to_mlir_type(sp_ndarray.row.dtype), format)
    else:
        raise Exception("Unsupported format")

def mlir_type_from_ndarray(A):
    if not sp.sparse.issparse(A):
        return tensor_type_from_dense_ndarray(A)
    else:
        return mlir_type_from_sparse_ndarray(A)

def mlir_type_from_python_type(value):
    if isinstance(value, int):
        mlir_type = 'index' 
    elif isinstance(value, float):
        mlir_type = 'f64' 
    else:
        raise Exception(f'Unsupported Python type {type(value)}')
    return mlir_type
        



def dtype_to_mlir_type(dtype):
    if dtype == 'int64':
        return 'i64'
    elif dtype == 'int32':
        return 'i32'
    elif dtype == 'float32':
        return 'f32'
    elif dtype == 'float64':
        return 'f64'
    else :
        raise Exception("YBE")
    
def mlir_type_to_dtype(mlir_type):
    if mlir_type == 'index':
        return 'int64'
    elif mlir_type == 'i64':
        return 'int64'
    elif mlir_type == 'i32':
        return 'int32'
    elif mlir_type == 'f32':
        return 'float32'
    elif mlir_type == 'f64':
        return 'float64'
    else :
        raise Exception("YBE")
    

def get_format(A):
    if not sp.sparse.issparse(A):
        return types.DENSE
    elif A.format == 'csr':
        return types.CSR
    elif A.format == 'coo':
        return types.COO
    elif A.format == 'csc':
        return types.CSC
    else:
        raise RuntimeError('Unsupported sparse format')


def format_to_string(format):
    if format == types.DENSE:
        return "Dense"
    elif format == types.CSR:
        return "CSR"
    elif format == types.COO:
        return "COO"
    else:
        raise RuntimeError('Unsupported sparse format')


def python_type_to_ctype(t):
    if isinstance(t, int):
        ctype = c_int32
    elif isinstance(t, float):
        ctype = c_double
    return ctype

def np_array_to_memref(np_array):
    ctype = c_int64
    if np_array.dtype == 'int32':
        ctype = c_int32
    elif np_array.dtype == 'float32':
        ctype = c_float
    elif np_array.dtype == 'float64':
        ctype = c_double
    return np_array.ctypes.data_as(POINTER(ctype)), np_array.ctypes.data_as(POINTER(ctype)), 0, np_array.shape[0], 1

def expand_memref_ptr(memref):
    return byref(memref), byref(memref), 0, 1, 1

def len_dense(vals):
    num = 0
    for v in vals:
        if not sp.sparse.issparse(v):
            num+=1
    return num


def memref_from_shaped_type(tensor_type):
    ctype = None
    shape = tensor_type.shape
    if tensor_type.element_type == 'index':
        ctype = c_int64
        if(len(shape) == 1):
            constructor = memref_i64
        else: 
            constructor = create_memref_type(ctype,len(shape))
    if tensor_type.element_type == 'i64':
        ctype = c_int64
        if(len(shape) == 1):
            constructor = memref_i64
        else: 
            constructor = create_memref_type(ctype,len(shape))
    elif tensor_type.element_type == 'i32':
        ctype = c_int32
        if(len(shape) == 1):
            constructor = memref_i32
        else: 
            constructor = create_memref_type(ctype,len(shape))
    elif tensor_type.element_type == 'f64':
        ctype = c_double
        if(len(shape) == 1):
            constructor = memref_f64
        else: 
            constructor = create_memref_type(ctype,len(shape))
    elif tensor_type.element_type == 'f32':
        ctype = c_float
        if(len(shape) == 1):
            constructor = memref_f32
        else: 
            constructor = create_memref_type(ctype,len(shape))
    return constructor(ctypes.cast(None, ctypes.POINTER(ctype)), ctypes.cast(None, ctypes.POINTER(ctype)), 0, (c_int64*len(shape))(*shape), (c_int64*len(shape))(*[0]*len(shape)))

def memref_from_np_array(np_array):
    ctype = None
    if np_array.dtype == 'int64':
        ctype = c_int64
        if(len(np_array.shape) == 1):
            constructor = memref_i64
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'int32':
        ctype = c_int32
        if(len(np_array.shape) == 1):
            constructor = memref_i32
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'float32':
        ctype = c_float
        if(len(np_array.shape) == 1):
            constructor = memref_f32
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'float64':
        ctype = c_double
        if(len(np_array.shape) == 1):
            constructor = memref_f64
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    if hasattr(np_array, '__cuda_array_interface__'):
        ptr = np_array.__cuda_array_interface__['data'][0]
        return constructor(ctypes.cast(ptr, ctypes.POINTER(ctype)), ctypes.cast(ptr, ctypes.POINTER(ctype)), 0, (c_int64*len(np_array.shape))(*np_array.shape), (c_int64*len(np_array.shape))(*[s//8 for s in np_array.strides]))
    else:
        return constructor(np_array.ctypes.data_as(ctypes.POINTER(ctype)), np_array.ctypes.data_as(ctypes.POINTER(ctype)), 0, (c_int64*len(np_array.shape))(*np_array.shape), (c_int64*len(np_array.shape))(*[s//8 for s in np_array.strides]))
