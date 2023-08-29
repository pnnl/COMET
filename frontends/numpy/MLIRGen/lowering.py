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
import atexit
import uuid
import scipy as scp
debug = False
class memref_i64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_longlong)), ('mem', POINTER(c_longlong)), ('offset', c_longlong), ('dim', c_longlong), ('stride', c_longlong)]

class memref_f64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_double)), ('mem', POINTER(c_double)), ('offset', c_longlong), ('dim', c_longlong), ('stride', c_longlong)]


def np_array_to_memref(np_array):
    ctype = ctypes.c_longlong
    if np_array.dtype == 'int32':
        ctype = c_int32
    elif np_array.dtype == 'float32':
        ctype = c_float
    elif np_array.dtype == 'float64':
        ctype = c_double
    return np_array.ctypes.data_as(ctypes.POINTER(ctype)), np_array.ctypes.data_as(ctypes.POINTER(ctype)), 0, np_array.shape[0], 1

def expand_memref_ptr(memref):
    return byref(memref), byref(memref), 0, 1, 1

def len_dense(vals):
    num = 0
    for v in vals:
        if not scp.sparse.issparse(v):
            num+=1
    return num

files_to_cleanup = []
def cleanup():
    for f in files_to_cleanup:
        if os.path.exists(f):
            os.remove(f) 
            # pass

atexit.register(cleanup)
if("macOS" in platform.platform()):
    comet_runner_util = "../build/lib/libcomet_runner_utils.dylib"
elif("Linux" in platform.platform()):
    comet_runner_util = "../build/lib/libcomet_runner_utils.so"
else:
    print("error: Support available only for Linux and macOS")
    sys.exit()

def lower_ta_to_mlir(mlir_in, mlir_lower_flags, uuid_s):

    # scf_out_file = 'einsum_loops.mlir'
    scf_out_file = uuid_s+'loops.mlir'

    if(os.path.exists(scf_out_file) == False):
        f = open(os.path.join( os.getcwd(), scf_out_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    else:
        f = open(scf_out_file, 'wb')

    path_to_comet = "../build/bin/comet-opt"

    command = path_to_comet + mlir_lower_flags + mlir_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))
    
    scf_out  = p.stderr
    f.write(scf_out)
    f.close()

    return scf_out_file

def all_dense(arg_vals) -> bool :
    for v in arg_vals:
        if scp.sparse.issparse(v):
            return False;
    return True

def comment_unneeded_dense(input_, arg_vals):
    input = input_.splitlines()
    out = None
    indexes = []
    for i, v in enumerate(arg_vals):
        if not scp.sparse.issparse(v):
            indexes.append(i)
    replace = {}
    fill_remove = []

    allocs_needed = len_dense(arg_vals)

    for i in range(len(input)):
        if "call @comet_print_memref_f64" in input[i]:
            cast = input[i][input[i].find("(") + 1 :  input[i].find(")")]
            # input[i]  = "// from dense " + input[i]
            input[i]  = ""
            for j in range(len(input[:i])):
                if cast + " = memref.cast" in input[j]:
                    out = input[j][input[j].find("%alloc")  : input[j].find(":")].lstrip().strip()
                    replace['%arg'+str(len(arg_vals))] = out
                    for k in range(len(input[:j])):
                        if out+" = memref.alloc(" in  input[k]:
                            # input[k] = "//from dense" + input[k]
                            input[k] = ""
        elif allocs_needed > 0 and "memref.alloc" in input[i]:
            allocs_needed = allocs_needed - 1
            a = input[i][input[i].find("%") : input[i].find("=")].lstrip().strip()
            replace['%arg'+str(indexes[0])] = a
            indexes = indexes[1:]
            # input[i] = "//from dense" + input[i]
            input[i] = ""
            fill_remove.append(a)
        elif "linalg.fill" in input[i]:
            for k in range(len(fill_remove)):
                if fill_remove[k] in input[i]:
                    # input[i] = "//from dense" + input[i]
                    input[i] = ""
                    fill_remove.remove(fill_remove[k])
                    break

    for v in replace:
        input[1] = input[1].replace(v, replace[v])
            
    output = ""

    for line in input:
        if line:
            output += line +"\n"

    return output            

def comment_unneeded_sparse(input_, arg_vals):
    output = ""
    input = input_.splitlines()
    indexes = []
    indexes = []
    allocs = []
    returns = []
    return_found = False
    out_len = len(input)
    for i in range(len(input)):
        line = input[i]

        if "call @read_input_sizes" in input[i]:
            cast = line[line.find("(") : line.find(")")].split(",")[3].lstrip().strip()
            # With tiles
            # cast = line[line.find("(") : line.find(")")].split(",")[5].lstrip().strip()
            # input[i] = "//" +input[i]
            input[i] = ""
            alloc = ""
            for j in range(len(input[:i])):
                lline = input[j]
                if cast +" = memref.cast" in lline:
                    alloc = lline.split()[3].lstrip().strip()
                    # input[j] = "// from sparse" + input[j]
                    # input[j] = ""
                    for k in range(len(input[:j])):
                        if alloc + " = memref.alloc" in input[k]:
                            # input[k] = "// from sparse"+input[k]
                            input[k] = ""
                            allocs.append(alloc)
            i+=1
            found = 0
            while(found != 7):
            # With tiles
            # while(found != 11):
                if "memref.load " + alloc in input[i]:
                    idx = input[i].split('=')[0].lstrip().strip()
                    indexes.append(idx)
                    found += 1
                i+=1
            idx = 0
        elif 'memref.alloc(' in input[i]:
            line = input[i]
            idx = line[line.find('(') + 1: line.find(')')]
            if idx in indexes[:-1]:
                indexes.remove(idx)
                allocs.append(line.split('=')[0].rstrip().strip())
                # input[i] = "// from sparse" + input[i]
                input[i] = ""
                while "scf.for" not in input[i]:
                    i+=1
                # input[i] = "// from sparse" + input[i]
                # input[i+1] = "// from sparse" + input[i+1]
                # input[i+2] = "// from sparse" + input[i+2]
                input[i] = ""
                input[i+1] = ""
                input[i+2] = ""
        elif "call @read_input_2D" in input[i]:
            # input[i] = '// from sparse' + input[i]
            input[i] = ""
        elif "call @comet_print_memref_i64" in input[i] or "call @comet_print_memref_f64" in input[i]:
            cast = input[i][input[i].find("(") + 1 : input[i].find(")")]
            for j in range(len(input[:i])):
                lline = input[j]
                if cast + " = memref.cast" in lline:
                    alloc = lline.split()[3].lstrip().strip()
                    returns.append((alloc, i))
        elif ("return" in input[i]) and len(returns) > 1 and not return_found:
            return_found = True
            add = ""
            for k, r in enumerate(returns[:-1]):
                add += "\t\tmemref.store {}, %marg{}[%c0] : memref<1xmemref<?xindex>>\n".format(r[0], k)
                # input[r[1]] = "//from sparse" + input[r[1]]
                input[r[1]] = ""
            # input[returns[-1][1]] = "//from sparse" + input[returns[-1][1]]
            input[returns[-1][1]] = ""
            add += "\t\tmemref.store {}, %marg{}[%c0] : memref<1xmemref<?xf64>>\n".format(returns[-1][0], len(returns)-1)
            add += "\t\treturn"
            input[i] = add

    args = input[1][input[1].find("(") + 1: input[1].find(")")].split(",")
    ai = 0
    for i, v in enumerate(arg_vals):
        if scp.sparse.issparse(v):
            input[1] = input[1].replace(args[i], allocs[ai] +" : memref<7xindex>, " + " : memref<?xindex>, ".join([s for s in allocs[ai+1:ai+6]]) + " : memref<?xf64>")
            # With tiles
            # input[1] = input[1].replace(args[i], allocs[ai] +" : memref<11xindex>, " + " : memref<?xindex>, ".join([s for s in allocs[ai+1:ai+10]]) + " : memref<?xf64>")
            ai += 6
            # With tiles 
            # ai += 10 
    if len(returns) > 1:
        input[1] = input[1].replace(")", ", %marg0: memref<1xmemref<?xindex>>, %marg1: memref<1xmemref<?xindex>>, %marg2: memref<1xmemref<?xindex>>, %marg3: memref<1xmemref<?xindex>>, %marg4: memref<1xmemref<?xf64>>)")
    input[-1] = '\n  func.func @dealloc(%to_dealloc: memref<?xindex>, %to_dealloc1: memref<?xindex>, %to_dealloc2: memref<?xindex>, %to_dealloc3: memref<?xindex>, %to_dealloc4: memref<?xf64>){\n \
\t\tmemref.dealloc %to_dealloc : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc1 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc2 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc3 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc4 : memref<?xf64>\n \
\t\treturn\n \
\t}\n\
}\n'
    # With tiles
    #         input[1] = input[1].replace(")", ", %marg0: memref<1xmemref<?xindex>>, %marg1: memref<1xmemref<?xindex>>, %marg2: memref<1xmemref<?xindex>>, %marg3: memref<1xmemref<?xindex>>, %marg4: memref<1xmemref<?xindex>>, %marg5: memref<1xmemref<?xindex>>, %marg6: memref<1xmemref<?xindex>>, %marg7: memref<1xmemref<?xindex>>, %marg8: memref<1xmemref<?xf64>>)")
#     input[-1] = '\n  func.func @dealloc(%to_dealloc: memref<?xindex>, %to_dealloc1: memref<?xindex>, %to_dealloc2: memref<?xindex>, %to_dealloc3: memref<?xindex>, %to_dealloc4: memref<?xindex>, %to_dealloc5: memref<?xindex>, %to_dealloc6: memref<?xindex>, %to_dealloc7: memref<?xindex>, %to_dealloc8: memref<?xf64>){\n \
# \t\tmemref.dealloc %to_dealloc : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc1 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc2 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc3 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc4 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc5 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc6 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc7 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc8 : memref<?xf64>\n \
# \t\treturn\n \
# \t}\n\
# }\n'
    for line in input:
        if line:
            output += line +"\n"
    return output

def lower_ta_to_mlir_with_jit(mlir_in, mlir_lower_flags, arg_vals, uuid_s):

    path_to_comet = "../build/bin/comet-opt"

    command = path_to_comet + mlir_lower_flags + mlir_in
    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if p.returncode != 0:
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error: {}".format(p.returncode, p.stderr.decode()))

    scf_out  = p.stderr.decode()
    scf_out = comment_unneeded_sparse(scf_out, arg_vals)
    scf_out = comment_unneeded_dense(scf_out, arg_vals)

    scf_out_file = uuid_s+'loops.mlir'

    if(os.path.exists(scf_out_file) == False):
        f = open(os.path.join( os.getcwd(), scf_out_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    else:
        f = open(scf_out_file, 'w')

    f.write(scf_out)
    f.close()

    return scf_out_file


def lower_scf_to_llvm(scf_in, scf_lower_flags, uuid_s):

    # llvm_out_file = 'einsum.llvm'

    # path_to_mliropt = "../llvm/build/bin/mlir-opt"
    path_to_cometopt = "../build/bin/comet-opt"

    command = path_to_cometopt + scf_lower_flags + scf_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))

    llvm_out_file = uuid_s+'.llvm'

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

def generate_llvm_args_from_ndarrays(num_in, *ndargs):
    llvm_args = []
    llvm_args_types = []
    all_outputs = []
    for i, ndarray in enumerate(ndargs):
        # Ndarray is dense
        if not scp.sparse.issparse(ndarray):
            llvm_args.append(ndarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            llvm_args.append(ndarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            llvm_args.append(0)
            llvm_args_types.append(ctypes.POINTER(ctypes.c_double))
            llvm_args_types.append(ctypes.POINTER(ctypes.c_double))
            llvm_args_types.append(ctypes.c_longlong)
            for s in ndarray.shape:
                llvm_args.append(s)
                llvm_args_types.append(ctypes.c_longlong)
            for s in ndarray.strides:
                llvm_args.append(s)
                llvm_args_types.append(ctypes.c_longlong)
            if i >= num_in:
                all_outputs.append(ndarray)
        # Ndarray is sparse
        else:
            if i >= num_in:
                A1pos = memref_i64()
                A1crd = memref_i64()
                A2pos = memref_i64()
                A2crd = memref_i64()
                Aval = memref_f64()

                llvm_args += [*expand_memref_ptr(A1pos), *expand_memref_ptr(A1crd), *expand_memref_ptr(A2pos), *expand_memref_ptr(A2crd), *expand_memref_ptr(Aval)]
                llvm_args_types += [POINTER(memref_i64), POINTER(memref_i64), c_longlong, c_longlong, c_longlong] * 4 + [POINTER(memref_f64), POINTER(memref_f64), c_longlong, c_longlong, c_longlong]
                all_outputs.append((A1pos, A1crd, A2pos, A2crd, Aval))
            else:
                # [TODO] The arrays used as inputs for the comet generated code need to be updated to take into account the extra tile component
                # CSR
                A1tile_pos = np.array([-1], dtype=np.int64)
                A1tile_crd = np.array([-1], dtype=np.int64)
                A2tile_pos = np.array([-1], dtype=np.int64)
                A2tile_crd = np.array([-1], dtype=np.int64)
                if scp.sparse.isspmatrix_csr(ndarray):
                    A1pos = np.array([ndarray.get_shape()[0]], dtype=np.int64)
                    A1crd = np.array([-1], dtype=np.int64)
                    A2pos = ndarray.indptr.astype('int64')
                    A2crd = ndarray.indices.astype('int64')

                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                    llvm_args += [*np_array_to_memref(np.array([1, 1, ndarray.get_shape()[0] + 1, ndarray.getnnz(), ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                    # With tiles
                    # llvm_args += [*np_array_to_memref(np.array([1, 1, 0, 0, ndarray.get_shape()[0] + 1, ndarray.getnnz(), 0, 0, ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                # COO
                elif scp.sparse.isspmatrix_coo(ndarray):
                    A1pos = np.array([0, ndarray.nnz], dtype=np.int64)
                    A1crd = ndarray.row.astype('int64')
                    A2pos = np.array([-1], dtype=np.int64)
                    A2crd = ndarray.col.astype('int64')

                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                    llvm_args += [*np_array_to_memref(np.array([2, ndarray.nnz, 1, ndarray.getnnz(), ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                    # With tiles
                    # llvm_args += [*np_array_to_memref(np.array([2, ndarray.nnz, 0, 0, 1, ndarray.getnnz(), 0, 0, ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                
                # CSC
                elif scp.sparse.isspmatrix_csc(ndarray):
                    A1pos = ndarray.indptr.astype('int64')
                    A1crd = ndarray.indices.astype('int64')
                    A2pos = np.array([ndarray.get_shape()[1]], dtype=np.int64)
                    
                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                    llvm_args += [*np_array_to_memref(np.array([ndarray.get_shape()[1] + 1, ndarray.nnz, 1, 1, ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                    # With tiles
                    # llvm_args += [*np_array_to_memref(np.array([ndarray.get_shape()[1] + 1, ndarray.nnz, 0, 0, 1, 1, 0, 0, ndarray.getnnz(), ndarray.get_shape()[0], ndarray.get_shape()[1]], dtype='int64'))]
                
                Aval = ndarray.data.astype('float64')

                # Based on the  desc_A1pos/crd, desc_A2pos/crd, desc_Aval arrays in SparseUtils.cpp: read_input_2D
                # Expand to memrefs llvmir implementation
                llvm_args += [*np_array_to_memref(A1pos),  *np_array_to_memref(A1crd), *np_array_to_memref(A2pos), *np_array_to_memref(A2crd), *np_array_to_memref(Aval)]
                # With tiles
                # llvm_args += [*np_array_to_memref(A1pos),  *np_array_to_memref(A1crd), *np_array_to_memref(A1tile_pos), *np_array_to_memref(A1tile_crd), *np_array_to_memref(A2pos), *np_array_to_memref(A2crd), *np_array_to_memref(A2tile_pos), *np_array_to_memref(A2tile_crd), *np_array_to_memref(Aval)]
                
                # Set the datatypes expected from the function in the shared library.
                # If we don't define this the data are not passed correctly
                llvm_args_types += [ctypes.POINTER(c_longlong), ctypes.POINTER(c_longlong), c_longlong, c_longlong, c_longlong] * 5  + [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_longlong, c_longlong, c_longlong]
                # With tiles
                # llvm_args_types += [ctypes.POINTER(c_longlong), ctypes.POINTER(c_longlong), c_longlong, c_longlong, c_longlong] * 9  + [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_longlong, c_longlong, c_longlong]

    return llvm_args, llvm_args_types, all_outputs

#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm_with_jit(llvm_in,func_name, inputs, outputs, uuid_s):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

    llvmir_out = p.stdout
    # llvmir_file = 'einsum.ll'
    llvmir_file = uuid_s+'.ll'
    libname = "lib"+llvmir_file+func_name+".so"
    if(os.path.exists(llvmir_file) == False):
        f = open(os.path.join( os.getcwd(), llvmir_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), llvmir_file))
    else:
        f = open(llvmir_file, 'wb')

    # with open(os.path.join( os.getcwd(),llvmir_file), 'wb') as f:
    f.write(llvmir_out)
    f.close()
    
    llc_command = "../llvm/build/bin/llc -O3 "+llvmir_file+" -filetype=obj -o " +llvmir_file+".o"
    p = subprocess.run(shlex.split(llc_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("lcc failed with error code: {}. Error: {}".format(p.returncode, p.stderr))
    files_to_cleanup.append(llvmir_file+ ".o")
    gcc_command = "gcc --shared  " +llvmir_file+ ".o -O3 -o "+libname+" -fpic -L ../build/lib/ -Wl,-rpath,../build/lib/ -lcomet_runner_utils"

    p = subprocess.run(shlex.split(gcc_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("gcc failed with error code: {}. Error: {}".format(p.returncode, p.stderr))

    # Load code generated from COMET
    lib = ctypes.cdll.LoadLibrary(libname)
    func = lib.__getattr__(func_name)
    
    # Get the inputs, input types and the output containers
    args, arg_types, all_output = generate_llvm_args_from_ndarrays(len(inputs),*(inputs), *(outputs))
    func.argtypes = arg_types

    # Uncomment to measure execution time without the compilation process
    
    # start = time.time()
    func(*(args))
    # end = time.time()
    # print("Kernel execution time JIT: {}".format(end-start))

    out = None
    ret_outputs = []
    for v0, v1 in zip(all_output, outputs):
        if scp.sparse.issparse(v1):
            A1pos, A1crd, A2pos, A2crd, Aval = v0
            if scp.sparse.isspmatrix_csr(v1):
                np_r_indices = np.ctypeslib.as_array(A2pos.mem, [A2pos.dim])
                np_c_indices = np.ctypeslib.as_array(A2crd.mem, [A2crd.dim])
                np_values = np.ctypeslib.as_array(Aval.mem, [Aval.dim])
                ret_outputs.append(scp.sparse.csr_matrix((np_values, np_c_indices, np_r_indices), copy=False))
            elif scp.sparse.isspmatrix_coo(v1):
                rows = np.ctypeslib.as_array(A1crd.mem, [A1crd.dim])
                cols = np.ctypeslib.as_array(A2crd.mem, [A2crd.dim])
                np_values = np.ctypeslib.as_array(Aval.mem, [Aval.dim])
                ret_outputs.append(scp.sparse.coo_matrix((np_values, (rows, cols)), copy=False))
        else:
            ret_outputs.append(v1)

    if len(ret_outputs) == 1:
        out =  ret_outputs.pop()
    else:
        out = ret_outputs
    files_to_cleanup.append(os.path.join( os.getcwd(), libname))

    return out, llvmir_file


def translate_and_exec_llvm(llvm_in,func_name, out_dims, uuid_s):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    # print(llvm_in)
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

def lower_dialect(ta_dialect_rep, out_dims, compile_with_flags,func_name):

    #lower TA dialect to the SCF dialect
    mlir_lower_flags = " --convert-ta-to-it --convert-to-loops "

    if isinstance(compile_with_flags,tuple):
        for i in range(len(compile_with_flags)):
            mlir_lower_flags += compile_with_flags[i] + " "
    
    elif(isinstance(compile_with_flags,str)):
        mlir_lower_flags += compile_with_flags + " "

    # scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "
    scf_lower_flags =  " --convert-to-llvm "

    if("-emit-ta" in mlir_lower_flags):
        print(ta_dialect_rep)
        return

     #write the TA dialect rep to file
    uuid_s = str(uuid.uuid4())
    ta_dialect_file = uuid_s+'.mlir'
    # print("uuid_s: ", uuid_s)
    if(os.path.exists(ta_dialect_file) == False):
        f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), ta_dialect_file))
    else:
        f = open(ta_dialect_file, 'w')
    
    f.write(ta_dialect_rep)
    f.close()

    # Uncomment for debugging pusposes
    scf_out_file = lower_ta_to_mlir(ta_dialect_file, mlir_lower_flags, uuid_s)
    # scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_file, mlir_lower_flags, args_vals)
    
    # Running --convert-ta-to-it --convert-to-loops and  --convert-to-llvm in separate steps 
    # does not produce correct output. This is an issue with the backend.

    #lower the SCF dialect to first STD dialect and then to the llvm dialect
    llvm_out_file = lower_scf_to_llvm(scf_out_file, scf_lower_flags, uuid_s)
    # llvm_out_file = lower_scf_to_llvm(ta_dialect_file, mlir_lower_flags + scf_lower_flags)
   
    result,llvmir_file = translate_and_exec_llvm(llvm_out_file,func_name, out_dims, uuid_s)
    
    return result


def lower_dialect_with_jit(ta_dialect_rep, out_dims, compile_with_flags,func_name, args_vals, outputs):
    #lower TA dialect to the SCF dialect
    mlir_lower_flags = " --convert-ta-to-it --convert-to-loops "

    if isinstance(compile_with_flags,tuple):
        for i in range(len(compile_with_flags)):
            mlir_lower_flags += compile_with_flags[i] + " "
    
    elif(isinstance(compile_with_flags,str)):
        mlir_lower_flags += compile_with_flags + " "

    # scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "
    scf_lower_flags =  " --convert-to-llvm "

    if("-emit-ta" in mlir_lower_flags):
        print(ta_dialect_rep)
        return

     #write the TA dialect rep to file
    # ta_dialect_file = 'einsum.mlir'
    uuid_s = str(uuid.uuid4())
    ta_dialect_file = uuid_s+'.mlir'
    if(os.path.exists(ta_dialect_file) == False):
        f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), ta_dialect_file))
    else:
        f = open(ta_dialect_file, 'w')
    
    f.write(ta_dialect_rep)
    f.close()

    # Uncomment for debugging pusposes
    scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_file, mlir_lower_flags, args_vals, uuid_s)
    
    # Running --convert-ta-to-it --convert-to-loops and  --convert-to-llvm in separate steps 
    # does not produce correct output. This is an issue with the backend.

    #lower the SCF dialect to first STD dialect and then to the llvm dialect
    llvm_out_file = lower_scf_to_llvm(scf_out_file, scf_lower_flags, uuid_s)
    # llvm_out_file = lower_scf_to_llvm(ta_dialect_file, mlir_lower_flags + scf_lower_flags)
   
    #lower the SCF dialect to the LLVM dialect
    #result = execute_llvm(llvm_out_file)

    result,llvmir_file = translate_and_exec_llvm_with_jit(llvm_out_file,func_name, args_vals, outputs, uuid_s)

    return result