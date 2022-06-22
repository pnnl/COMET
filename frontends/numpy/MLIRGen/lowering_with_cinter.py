

import shlex
import ctypes
import numpy as np
import os,subprocess
import time

 # Load the shared library into ctypes
lib_bridge = "../build/numpy/core/libcomet-mlir.dylib"

c_bridge = ctypes.cdll.LoadLibrary(lib_bridge)


def lower_dialect(ta_dialect_rep:str, out_dims, compile_with_flags,func_name):

    ta_dialect_rep = ta_dialect_rep.replace(func_name, "main")

    ta_char_stream = ctypes.c_char_p(ta_dialect_rep.encode('utf-8'))

    c_bridge.comet_mlir(ta_char_stream)

    llvmir_file = 'einsum.ll'

    lli_command = "../llvm/build/bin/lli -O3 -load ../build/lib/libcomet_runner_utils.dylib " + llvmir_file

    start = time.perf_counter()
    p = subprocess.run(shlex.split(lli_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    end = time.perf_counter()
 
    print("Exec time: " , end-start, " s")

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

    os.remove(llvmir_file)
    if(len(output_arrays_list) > 1):
        return tuple(output_arrays_list)
    else:
        return output_arrays_list.pop()



    
        
    


  


