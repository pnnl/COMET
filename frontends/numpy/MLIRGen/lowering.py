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

if("macOS" in platform.platform()):
    comet_runner_util = "../build/lib/libcomet_runner_utils.dylib"
elif("Linux" in platform.platform()):
    comet_runner_util = "../build/lib/libcomet_runner_utils.so"
else:
    print("error: Support available only for Linux and macOS")
    sys.exit()

def lower_ta_to_mlir(mlir_in, mlir_lower_flags):

    scf_out_file = 'einsum_loops.mlir'

    if(os.path.exists(scf_out_file) == False):
        f = open(os.path.join( os.getcwd(), scf_out_file), 'wb')
    else:
        f = open(scf_out_file, 'wb')

    path_to_comet = "../build/bin/comet-opt"

    command = path_to_comet + mlir_lower_flags + mlir_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)

    scf_out  = p.stderr
    f.write(scf_out)
    f.close()

    return scf_out_file


def lower_scf_to_llvm(scf_in, scf_lower_flags):

    llvm_out_file = 'einsum.llvm'

    if(os.path.exists(llvm_out_file) == False):
        f = open(os.path.join( os.getcwd(), llvm_out_file), 'wb')
    else:
        f = open(llvm_out_file, 'wb')

    path_to_mliropt = "../llvm/build/bin/mlir-opt"

    command = path_to_mliropt + scf_lower_flags + scf_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

    llvm_out = p.stdout

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


#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm(llvm_in,func_name):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

    llvmir_out = p.stdout.decode()

    llvmir_out = llvmir_out.replace(func_name, "main")

    llvmir_file = 'einsum.ll'

    with open(os.path.join( os.getcwd(),llvmir_file), 'w') as f:
        f.write(llvmir_out)

    lli_command = "../llvm/build/bin/lli -load " + comet_runner_util  + " einsum.ll"
    
    p = subprocess.run(shlex.split(lli_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    result = p.stdout
        
    return result,llvmir_file


def lower_dialect(ta_dialect_rep, out_dims, compile_with_flags,func_name):

    #lower TA dialect to the SCF dialect
    mlir_lower_flags = " --convert-ta-to-it --convert-to-loops "

    if isinstance(compile_with_flags,tuple):
        for i in range(len(compile_with_flags)):
            mlir_lower_flags += compile_with_flags[i] + " "
    
    elif(isinstance(compile_with_flags,str)):
        mlir_lower_flags += compile_with_flags + " "

    scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "

    if("-emit-ta" in mlir_lower_flags):
        print(ta_dialect_rep)
        return

     #write the TA dialect rep to file
    ta_dialect_file = 'einsum.mlir'
    if(os.path.exists(ta_dialect_file) == False):
        f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
    else:
        f = open(ta_dialect_file, 'w')
    
    f.write(ta_dialect_rep)
    f.close()

    scf_out_file = lower_ta_to_mlir(ta_dialect_file, mlir_lower_flags)
  
    #lower the SCF dialect to first STD dialect and then to the llvm dialect
    llvm_out_file = lower_scf_to_llvm(scf_out_file, scf_lower_flags)
   
    #lower the SCF dialect to the LLVM dialect
    #result = execute_llvm(llvm_out_file)

    result,llvmir_file = translate_and_exec_llvm(llvm_out_file,func_name)

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

    os.remove(ta_dialect_file)
    os.remove(scf_out_file)
    os.remove(llvm_out_file)
    os.remove(llvmir_file)

    if(len(output_arrays_list) > 1):
        return tuple(output_arrays_list)
    else:
        return output_arrays_list.pop()



    
        
    


  


