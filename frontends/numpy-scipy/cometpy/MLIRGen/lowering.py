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

import os
import subprocess
import shlex
import numpy as np
import platform
import sys
import time
import ctypes
from ctypes import *
import cometpy.cfg as cfg
from cometpy.MLIRGen import types
from cometpy.MLIRGen import ops
import atexit
import uuid
import scipy as scp
from cometpy.MLIRGen.utils import *

debug = False
temp_dir = '.cometpy/'
class KernelCache:

    def __init__(self):
        self.map = {}

    def find(self, func_name, input_types):
        cached_kernels = self.map.get(func_name)
        if cached_kernels:
            cached_kernel = cached_kernels.get("_".join(input_types))
            if not cached_kernel:
                return None
            if cached_kernel and cached_kernel.timestamp <  os.path.getmtime(func_name.split(':')[0]): 
                return  None
            return cached_kernel
        else:
            return None
    
    def insert(self, func_name, cached_kernel):
        cached_kernels = self.map.get(func_name)
        if cached_kernels:
            cached_kernels["_".join(cached_kernel.input_types)]  = cached_kernel
        else:
            self.map[func_name] = { "_".join(cached_kernel.input_types) : cached_kernel}

cache = KernelCache()

class CachedKernelInfo:

    def __init__(self, name, src_path, lib_path, timestamp, kernel_name, input_types, output_types):
        self.name = name
        self.src_path = src_path
        self.lib_path = lib_path
        self.timestamp = timestamp
        self.kernel_name = kernel_name
        self.input_types = input_types
        self.output_types = output_types


if not os.path.exists(temp_dir):
    try:
        os.mkdir(temp_dir)
    except:
        if os.path.exists(temp_dir):
            pass
        else:
            raise "Could not create .cometpy/"

platform_args = ""


if("macOS" in platform.platform()):
    comet_runner_util = cfg.comet_path+"/lib/libcomet_runner_utils.dylib"
    platform_args = "-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/ "
elif("Linux" in platform.platform()):
    comet_runner_util = cfg.comet_path+"/lib/libcomet_runner_utils.so"
else:
    print("error: Support available only for Linux and macOS")
    sys.exit()
files_to_cleanup = []
def cleanup():
    for f in files_to_cleanup:
        if os.path.exists(f):
            os.remove(f)
            # pass
atexit.register(cleanup)




def lower_ta_to_mlir(mlir_in, mlir_lower_flags, uuid_s):

    # scf_out_file = 'einsum_loops.mlir'
    scf_out_file = uuid_s+'loops.mlir'

    if(os.path.exists(scf_out_file) == False):
        f = open(os.path.join( os.getcwd(), scf_out_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    else:
        f = open(scf_out_file, 'wb')

    path_to_comet = cfg.comet_path+"/bin/comet-opt"

    command = path_to_comet + mlir_lower_flags + mlir_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))
    
    scf_out  = p.stderr
    f.write(scf_out)
    f.close()

    return scf_out_file

def lower_ta_to_mlir_with_jit(mlir_in, mlir_lower_flags, arg_vals, uuid_s):
    path_to_comet = cfg.comet_path+"/bin/comet-opt -x mlir "
    command = path_to_comet + mlir_lower_flags
    p = subprocess.run(shlex.split(command), input = mlir_in.encode('utf-8') , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if p.returncode != 0:
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error: {}".format(p.returncode, p.stderr.decode()))


    # scf_out_file = temp_dir + uuid_s+'loops.mlir'

    # if(os.path.exists(scf_out_file) == False):
    #     f = open(os.path.join( os.getcwd(), scf_out_file), 'w')
    #     files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    # else:
    #     f = open(scf_out_file, 'w')
    #     files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))

    scf_out  = p.stderr.decode()
    # scf_out = comment_unneeded_sparse(scf_out, arg_vals)
    # scf_out = comment_unneeded_dense(scf_out, arg_vals)
    # f.write(scf_out)
    # f.close()

    return scf_out


def lower_scf_to_llvm(scf_in, scf_lower_flags, uuid_s):

    # llvm_out_file = 'einsum.llvm'

    # path_to_mliropt = "../llvm/build/bin/mlir-opt"
    path_to_cometopt = cfg.comet_path+"/bin/comet-opt"

    command = path_to_cometopt + scf_lower_flags + scf_in
    # command = path_to_cometopt + scf_lower_flags + ' 38ee2ea1-6e20-439f-b7c0-73d7fa6b7da5loops.mlir'

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))

    llvm_out_file = temp_dir + uuid_s+'.llvm'

    if(os.path.exists(llvm_out_file) == False):
        f = open(os.path.join( os.getcwd(), llvm_out_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), llvm_out_file))
    else:
        f = open(llvm_out_file, 'wb')

    llvm_out = p.stderr

    f.write(llvm_out)
    f.close()

    return llvm_out_file

# def execute_llvm(llvm_in,func_name):
    
#     path_to_mlir_cpu_runner = "../llvm/build/bin/mlir-cpu-runner "

#     flags = ' -O3 -e ' + func_name + ' -entry-point-result=void -shared-libs=../llvm/build/lib/libmlir_runner_utils.dylib,../build/lib/libcomet_runner_utils.dylib'

#     command = path_to_mlir_cpu_runner + llvm_in + flags

#     p = subprocess.run(shlex.split(command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

#     result = p.stdout
        
#     return result
def memref_index_type():
    return [ctypes.POINTER(c_int64), ctypes.POINTER(c_int64), c_int64, c_int64, c_int64]

def memref_f64_type():
    return [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_int64, c_int64, c_int64]

def generate_llvm_args_from_ndarrays(inputs, output_types):
    llvm_args_types_len = 0 
    for out_type in output_types:
        if isinstance(out_type, types.ShapedType):
            llvm_args_types_len += 1
    llvm_args_len = llvm_args_types_len
    for input in inputs:
        if scp.sparse.issparse(input):
            llvm_args_len += 7
        else:
            llvm_args_len += 1

    llvm_args = [None] * llvm_args_len
    aux = [] # Used to make sure data created in this function are not freed prematurely.
    
    for i, out_type in enumerate(output_types):
        if isinstance(out_type, types.ShapedType):
            if out_type.format == types.DENSE:
                memref = memref_from_shaped_type(out_type)
                llvm_args[i] = memref
            else:
                dim_sizes = memref_from_shaped_type(types.TensorType([1], 'index') )
                aux.append(dim_sizes)
                Aval = memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.element_type))
                if out_type.format == types.CSR:
                    A1pos_temp = np.array([out_type.shape[0]], dtype= ops.mlir_type_to_dtype(out_type.indices_type))
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos = memref_from_np_array(A1pos_temp) #memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))
                    A2pos = memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))
                    A2crd = memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))


                    if out_type.element_type == 'f64':
                        if out_type.indices_type == 'i32':
                            out_ctor = output_csr_f64_i32
                        else:
                            out_ctor = output_csr_f64_i64
                    elif out_type.element_type == 'f32':
                        if out_type.indices_type == 'i32':
                            out_ctor = output_csr_f32_i32
                        else:
                            out_ctor = output_csr_f32_i64

                    out = out_ctor(dim_sizes, 0, A1pos, 0, A2pos, A2crd, Aval)
                elif out_type.format == types.COO:
                    A1pos_temp = np.array([0, 1], dtype=ops.mlir_type_to_dtype(out_type.indices_type))
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos =  memref_from_np_array(A1pos_temp) #memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))
                    A1crd =  memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))
                    A2crd =  memref_from_shaped_type(types.TensorType([out_type.shape[0]], out_type.indices_type))

                    if out_type.element_type == 'f64':
                        if out_type.indices_type == 'i32':
                            out_ctor = output_coo_f64_i32
                        else:
                            out_ctor = output_coo_f64_i64
                    elif out_type.element_type == 'f32':
                        if out_type.indices_type == 'i32':
                            out_ctor = output_coo_f32_i32
                        else:
                            out_ctor = output_coo_f32_i64

                    out = out_ctor(dim_sizes, 0, A1pos, A1crd, 0, A2crd, Aval)
                else :
                    raise Exception("Unsupported sparse matrix type")

                llvm_args[i] = out
    offset = llvm_args_types_len
    for i, ndarray in enumerate(inputs):
        

        if not scp.sparse.issparse(ndarray):
            # ndarray = np.array(out_type.shape, dtype=ops.mlir_type_to_dtype(out_type.element_type)) ## [TODO] No need to allocate
            if isinstance(ndarray, np.ndarray):
                memref = memref_from_np_array(ndarray)
                llvm_args[offset + i] = memref
            else: 
                llvm_args[offset + i] = ndarray
        else:
            dims = np.array(ndarray.shape, dtype=np.int64)
            dim_sizes = memref_from_np_array(dims)
            aux.append(dims)
            Aval = memref_from_np_array(ndarray.data) # memref_from_shaped_type(types.TensorType([ndarray.shape[0]], out_type.element_type))
            insert_pos_1 = 0
            insert_pos_2 = 0
            # CSR
            if ndarray.format == 'csr':
                # A1tile_pos = np.array([-1], dtype=ndarray.indptr.dtype)
                # A1tile_crd = np.array([-1], dtype=ndarray.indptr.dtype)
                # A2tile_pos = np.array([-1], dtype=ndarray.indptr.dtype)
                # A2tile_crd = np.array([-1], dtype=ndarray.indptr.dtype)
                A1pos_temp = np.array([ndarray.shape[0]], dtype=ndarray.indptr.dtype)
                aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                A1pos = memref_from_np_array(A1pos_temp)
                A2pos = memref_from_np_array(ndarray.indptr)
                A2crd = memref_from_np_array(ndarray.indices)

                # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                llvm_args[offset + i] = dim_sizes
                offset += 1 
                llvm_args[offset + i] = insert_pos_1
                offset += 1 
                llvm_args[offset + i] = A1pos
                offset += 1 
                llvm_args[offset + i] = insert_pos_2
                offset += 1 
                llvm_args[offset + i] = A2pos
                offset += 1 
                llvm_args[offset + i] = A2crd
                offset += 1 
            # COO
            elif ndarray.format == 'coo':
                # A1tile_pos = np.array([-1], dtype=ndarray.row.dtype)
                # A1tile_crd = np.array([-1], dtype=ndarray.row.dtype)
                # A2tile_pos = np.array([-1], dtype=ndarray.row.dtype)
                # A2tile_crd = np.array([-1], dtype=ndarray.row.dtype)
                A1pos_temp = np.array([0, ndarray.nnz], dtype=ndarray.row.dtype)
                aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                A1pos = memref_from_np_array(A1pos_temp)
                A1crd = memref_from_np_array(ndarray.row)
                A2crd = memref_from_np_array(ndarray.col)
                llvm_args[offset + i] = dim_sizes
                offset += 1 
                llvm_args[offset + i] = insert_pos_1
                offset += 1 
                llvm_args[offset + i] = A1pos
                offset += 1 
                llvm_args[offset + i] = A1crd
                offset += 1 
                llvm_args[offset + i] = insert_pos_2
                offset += 1 
                llvm_args[offset + i] = A2crd
                offset += 1
            # # CSC
            # elif ndarray.format == 'csc':
            #     A1pos = ndarray.indptr.astype('int64')
            #     A1crd = ndarray.indices.astype('int64')
            #     A2pos = np.array([ndarray.shape[1]], dtype=np.int64)
                
            #     # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
            
            #     # llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 1, 1, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
            #     # With tiles
            #     llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 0, 0, 1, 1, 0, 0, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
            
            # Based on the  desc_A1pos/crd, desc_A2pos/crd, desc_Aval arrays in SparseUtils.cpp: read_input_2D
            # Expand to memrefs llvmir implementation
            llvm_args[offset + i] = Aval
    return llvm_args, aux

#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm_with_jit(llvm_in,scf_lower_flags, kernel_name, inputs, output_types, uuid_s, func_name, input_types, cached_kernel):
    llvmir_file = None
    if not cached_kernel:
        llvmir_file = uuid_s+'.ll'
        llvm_in = llvm_in.replace('call @comet_print_memref_', '//call @comet_print_memref_')
        # path_to_cometopt = cfg.comet_path+"/bin/comet-opt"
        path_to_cometopt = cfg.comet_path+"/bin/comet-opt -x mlir"
        to_llvm_command = path_to_cometopt + scf_lower_flags #+ llvm_in
        translate_mlir_command = cfg.llvm_path+"/bin/mlir-translate --mlir-to-llvmir -- " 
        libname =   "./lib"+llvmir_file+kernel_name+".so"
        gcc_command = cfg.llvm_path+"/bin/clang -march=native -mtune=native -x ir -Wno-everything --shared -O3 "+platform_args+ " -o "+ temp_dir + libname+" -fpic -L {0}/lib/ -Wl,-rpath,{0}/lib/ -lcomet_runner_utils -L {1}/lib/ -Wl,-rpath,{1}/lib/ -fopenmp -".format(cfg.comet_path, cfg.llvm_path)

        # We merge all several calls in a single call to the shell in order to only pay the overhead of process creation once.
        # 1. Call comet to lower scf code to llvm
        # 2. Call mlir-translate to convert llvm to llvmir 
        # 3. Call clang to generate library
        # p = subprocess.run(to_llvm_command, input=llvm_in.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # start = time.time()
        p = subprocess.run(to_llvm_command +' 2>&1 |  '+ translate_mlir_command +' | ' + gcc_command , input=llvm_in.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if(p.returncode != 0):
            cleanup()
            raise AssertionError("gcc failed with error code: {}. Error: {} {}".format(p.returncode, p.stdout, p.stderr))
        files_to_cleanup.append(os.path.join( os.getcwd(), temp_dir +libname))
        # end = time.time()
        # print(f'To llvm time: {end-start}')
        # start = time.time()

        # Load code generated from COMET
        lib = ctypes.cdll.LoadLibrary(temp_dir +libname)
        func = lib.__getattr__("_mlir_ciface_"+kernel_name)
        timestamp = os.path.getmtime(temp_dir +libname)

        # Get the inputs, input types and the output containers
        args, aux = generate_llvm_args_from_ndarrays(inputs, output_types)
        # func.argtypes = arg_types
        
        # end = time.time()
        # print(f'dlopen time: {end-start}')
                
        # Uncomment to measure execution time without the compilation process
        # start = time.time()
        cache.insert(func_name, CachedKernelInfo(func_name, func_name.split(':')[0], func, timestamp, kernel_name, input_types, output_types))
    else:
        args, aux = generate_llvm_args_from_ndarrays(inputs, cached_kernel.output_types)
        output_types = cached_kernel.output_types
        func = cached_kernel.lib_path


    if len(output_types) == 1:
            if not isinstance(output_types[0], types.ShapedType):
                type_str = str(output_types[0])
                if type_str == 'i64':
                    func.restype = c_int64
                elif type_str == 'i32':
                    func.restype = c_int32
                elif type_str == 'f32':
                    func.restype = c_float
                elif type_str == 'f64':
                    func.restype = c_double
                else:
                    raise Exception("Unexpeted return type")
    # end = time.time()
    # print("create args: {}".format(end-start))
    # start = time.time()
    ret = func(*[byref(arg) if not isinstance(arg, int) else arg for arg in args])
    # end = time.time()
    # print("Kernel execution time JIT: {}".format(end-start))

    out = None
    ret_outputs = []
    for v0, v1 in zip(args, output_types):
        if isinstance(v1, types.ShapedType):
            if v1.format == types.CSR:
                out_csr = v0
                np_r_indices = np.ctypeslib.as_array(out_csr.A2pos.mem, [out_csr.A2pos.dim[0]])
                np_c_indices = np.ctypeslib.as_array(out_csr.A2crd.mem, [out_csr.A2crd.dim[0]])
                np_values = np.ctypeslib.as_array(out_csr.Aval.mem, [out_csr.Aval.dim[0]])
                ret_outputs.append(scp.sparse.csr_array((np_values, np_c_indices, np_r_indices), copy=False))
            elif v1.format == types.COO:
                out_coo = v0
                rows = np.ctypeslib.as_array(out_coo.A1crd.mem, [out_coo.A1crd.dim[0]])
                cols = np.ctypeslib.as_array(out_coo.A2crd.mem, [out_coo.A2crd.dim[0]])
                np_values = np.ctypeslib.as_array(out_coo.Aval.mem, [out_coo.Aval.dim[0]])
                ret_outputs.append(scp.sparse.coo_array((np_values, (rows, cols)), copy=False))
            elif v1.format == types.DENSE:
                ret_outputs.append(np.ctypeslib.as_array(v0.mem, v1.shape))
    
    if output_types and not ret_outputs and  not isinstance(output_types[0], types.ShapedType):
        ret_outputs.append(ret)
    
    if len(ret_outputs) == 1:
        out =  ret_outputs.pop()
    elif len(ret_outputs) > 1:
        out = ret_outputs
    else:
        out = None
    # print("Kernel execution time JIT: {}".format(end-start))
    return out, llvmir_file

def translate_and_exec_llvm(llvm_in,func_name, out_dims, uuid_s):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    llvmir_out = p.stdout.decode()

    llvmir_out = llvmir_out.replace(func_name, "main")

    llvmir_file = uuid_s+'.ll'

    if(os.path.exists(llvmir_file) == False):
        f = open(os.path.join( os.getcwd(), llvmir_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), llvmir_file))
    else:
        f = open(llvmir_file, 'w')

    # with open(os.path.join( os.getcwd(),llvmir_file), 'wb') as f:
    f.write(llvmir_out)
    f.close()

    # lli_command = "../llvm/build/bin/lli -load " + comet_runner_util  + " "+ llvmir_file
    lli_command = "../llvm/build/bin/mlir-cpu-runner " + llvm_in + " -O3 -e "+func_name+" -entry-point-result=void -shared-libs=../build/lib/libcomet_runner_utils.dylib"
    # print(lli_command)
    start = time.time()
    p = subprocess.run(shlex.split(lli_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    result = p.stdout
    result_str = result.decode("ascii").replace("\n","").strip().split("data =")

    output_arrays_list = []
    indx = 0
    for str_out in result_str:
        str_out = str_out.strip()
        output_array = np.fromstring(str_out, dtype=float, sep=',')
        
        if(output_array.size > 0):
            output_array = output_array.reshape(tuple(out_dims[indx]))
            output_arrays_list.append(output_array)
            indx = indx + 1
    end = time.time()
    print("Kernel execution time no JIT: {}".format(end-start))

    if(len(output_arrays_list) > 1):
        return tuple(output_arrays_list),llvmir_file
    else:
        return output_arrays_list.pop(),llvmir_file

def lower_dialect_with_jit(ta_dialect_rep, target: str, out_dims, compile_with_flags,kernel_name, args_vals, outputs_types, func_name, input_types, cached_kernel):

    scf_lower_flags = None
    scf_out_file = None
    uuid_s = None
    if not cached_kernel:
        mlir_lower_flags = " "
            
        if compile_with_flags != None:
            if "--convert-tc-to-ttgt" not in compile_with_flags:
                mlir_lower_flags += "  --convert-ta-to-it "
            if "--opt-fusion"  in compile_with_flags:
                mlir_lower_flags += "--opt-fusion"
                compile_with_flags = compile_with_flags.replace("--opt-fusion","")
                compile_with_flags = compile_with_flags.replace("--opt-comp-workspace","")
            # if "-opt-matmul-tiling" not in compile_with_flags:
            mlir_lower_flags += "   --convert-to-loops "
            mlir_lower_flags =" "+compile_with_flags + mlir_lower_flags
        else:
            mlir_lower_flags = "  --convert-ta-to-it --convert-to-loops "
        # scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "
        scf_lower_flags =  " --convert-to-llvm "
        
        if target != "cpu":
            if target.startswith("sm_") or target.startswith("compute_") or target.startswith("lto_"):
                if not cfg.gpu_target_enabled:
                    raise Exception("COMET gpu target is not enabled")
                scf_lower_flags += " " + "--convert-to-triton --target=GPU --gpu-compute-capability="+target.split("_")[1]
                mlir_lower_flags += " " + "--target=GPU"
            elif target == "gpu":
                if not cfg.gpu_target_enabled:
                    raise Exception("COMET gpu target is not enabled")
                scf_lower_flags += " " + "--convert-to-triton --target=GPU"
                mlir_lower_flags += " " + "--target=GPU"
            else :
                raise "Expected target formats:\
                        cpu, compute_<version>, sm_<version>, lto_<version>"
        
        if("-emit-ta" in mlir_lower_flags):
            print(ta_dialect_rep)
            return

        uuid_s = str(uuid.uuid4())
        scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_rep, mlir_lower_flags, args_vals, uuid_s)
    # end = time.time()
    # print(f"To SCF time: {end-start}")

    #lower the SCF dialect to LLVMIR and execute
    result,llvmir_file = translate_and_exec_llvm_with_jit(scf_out_file, scf_lower_flags, kernel_name, args_vals, outputs_types, uuid_s, func_name, input_types, cached_kernel)

    return result