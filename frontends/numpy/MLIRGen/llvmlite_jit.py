
from __future__ import print_function

from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm

import os
import subprocess
import shlex

import numpy as np



# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

llvm.load_library_permanently("../build/lib/libcomet_runner_utils.dylib")


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


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


#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm(llvm_in):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

    llvmir_out = p.stdout.decode()

    llvmir_out = llvmir_out.replace("@einsum", "@main")


    return llvmir_out



def lower_dialect(ta_dialect_rep, out_dims, compile_with_flags):

    #write the TA dialect rep to file
    ta_dialect_file = 'einsum.mlir'
    if(os.path.exists(ta_dialect_file) == False):
        f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
    else:
        f = open(ta_dialect_file, 'w')
    
    f.write(ta_dialect_rep)
    f.close()

    #lower TA dialect to the SCF dialect
    mlir_lower_flags = ""

    if compile_with_flags:
        for i in range(len(compile_with_flags)):
            mlir_lower_flags += compile_with_flags[i]

    else:
        mlir_lower_flags = " --convert-ta-to-loops "


    scf_lower_flags =  " --convert-scf-to-std --convert-std-to-llvm "

    scf_out_file = lower_ta_to_mlir(ta_dialect_file, mlir_lower_flags)
  
    #lower the SCF dialect to first STD dialect and then to the llvm dialect
    llvm_out_file = lower_scf_to_llvm(scf_out_file, scf_lower_flags)
   
    #Convert LLVM dialect to LLVM IR and JIT execute

    llvm_ir = translate_and_exec_llvm(llvm_out_file)

    engine = create_execution_engine()
    mod = compile_ir(engine, llvm_ir)

    # Look up the function pointer (a Python int)
    func_ptr = engine.get_function_address("main")

    # Run the function via ctypes
    cfunc = CFUNCTYPE(c_double)(func_ptr)

    result_str = str(cfunc()).replace("\n","").strip().split("data =")

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

    if(len(output_arrays_list) > 1):
        return tuple(output_arrays_list)
    else:
        return output_arrays_list.pop()



    
        