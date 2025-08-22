import shutil
import subprocess
import sys
import argparse
import os
from pathlib import Path

libspir_hls = 'libspir64-39-hls.bc'
libsqlite = 'libsqlite3.28.0.so'
clang_version = '3.9'

def retrieve_xilinx_paths():
    if os.getenv('VPP_PATH'):
        vpp_path = os.getenv('VPP_PATH')
    else :
        vpp_path = shutil.which('v++')
    
    if vpp_path is None:
        raise RuntimeError('Path to v++ not found. Please set the environmental variable "VPP_PATH" to the v++ executable location')
    
    path = Path(vpp_path)
    vpp_path = path.parents[0]
    version = path.parents[1].name
    vitis_hls_path = path.parents[3].joinpath('Vitis_HLS', version)
    
    # Usually in ../Xilinx/Vits_HLS/<version>/lnx64/lib/
    libspir_hls_path = vitis_hls_path.joinpath('lnx64', 'lib', f'{libspir_hls}')
    
    if not libspir_hls_path.exists() :
        libspir_hls_path = os.getenv('LIBSPIR_HLS_PATH')
        if libspir_hls_path is None:
            raise RuntimeError(f'Path to libspir-*-hls.bc not found')
    
    # Usually in ../Xilinx/Vits_HLS/<version>/lib/lnx64.o/lib/
    libsqllite_path = vitis_hls_path.joinpath('lib', 'lnx64.o')

    if not libsqllite_path.joinpath(libsqlite).exists() :
        libsqllite_path = os.getenv('LIBSQLLITE_PATH')
        if libsqllite_path is None:
            raise RuntimeError(f'Path to libsqlite not. Please set the environmental variable "LIBSQLLITE_PATH" to the directory containing it')
    
    llvm_hls_path = vitis_hls_path.joinpath('lnx64', 'tools', f'clang-{clang_version}-csynth', 'bin')

    # Usually in ../Xilinx/Vits_HLS/<version>/lnx64/tools/clang-<version>-csynth/bin/
    if not llvm_hls_path.joinpath('llvm-as').exists() :
        llvm_hls_path = os.getenv('LLVM_HLS_BIN_PATH')
        if llvm_hls_path is None:
            raise RuntimeError(f'Path to Xilinx llvm binaries not found. Please set the environmental variable "LLVM_HLS_BIN_PATH to the directory containing them')

    # Usually in ../Xilinx/Vits_HLS/<version>/lnx64/tools/clang-<version>-csynth/bin/
    if not llvm_hls_path.joinpath('llvm-link').exists() :
        raise RuntimeError(f'Path to Xilinx llvm-link not found')
    
    return (libspir_hls_path, libsqllite_path, llvm_hls_path, vpp_path)

llvm_spirv_path = '/qfs/people/thom895/TwoTruths/COMET/tools/llvm-spirv/build/bin/'
libspir_hls_path, lib_sqllite_path, llvm_hls_path, vpp_path = retrieve_xilinx_paths()
platform = 'xilinx_vck5000_gen3x16_xdma_1_202120_1'

def convert_ocl_ids_storage_class(input, output):
    ret = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', '-to-text', '-o', f'{output}.spt', input])
    if ret != 0 :
        print('Error in converting SPIRV binary to text')

    lines = None
    with open(f'{output}.spt', 'r') as f:
        vars = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'BuiltIn 26' in line or 'BuiltIn 27' in line or 'BuiltIn 28' in line:
                if 'Decorate ' in line:
                    vars.append(line.split()[2])
            elif '4 Variable 1' in line and line.strip().endswith('1'):
                lines[i] = line.strip()[:-1]+'0\n' 

    with open(f'{output}.spt', 'w') as f:
        f.writelines(lines)

    ret = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', '-to-binary', '-o', f'{output}.spv', f'{output}.spt'])
    if ret != 0 :
        print('Error in converting modified SPIRV text to binary')

def generate_xclbin(input, output, kernel, platform):
    
    convert_ocl_ids_storage_class(input, output)

    ret = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', '-r', '-o', f'{output}.bc', f'{output}.spv'])
    if ret != 0 :
        print('Error in converting SPIRV binary to bitcode')
        
    ret = subprocess.call([f'{llvm_spirv_path}/llvm-dis', '-o', f'{output}.ll', f'{output}.bc'])
    if ret != 0 :
        print('Error in converting SPIRV bitcode to LLVMIR')
        
    my_env = os.environ.copy()
    if 'LD_LIBRARY_PATH' in my_env:
        my_env['LD_LIBRARY_PATH'] = f"{lib_sqllite_path}:{my_env['LD_LIBRARY_PATH']}"
    else :
        my_env['LD_LIBRARY_PATH'] = f'{lib_sqllite_path}'
    print(my_env['LD_LIBRARY_PATH'])

    ret = subprocess.call([f'{llvm_hls_path}/llvm-as', '-o', f'{output}.xpirbc', f'{output}.ll'], env=my_env)
    if ret != 0 :
        print('Error in converting SPIRV LLVIR to .xpirbc')
        return
        
    ret = subprocess.call([f'{llvm_hls_path}/llvm-link', '-o', f'{output}.linked.xpirbc', f'{output}.xpirbc', f'{libspir_hls_path}'], env=my_env)
    if ret != 0 :
        print(f'Error in linking {output}.xpirbc ')
        return
    
    ret = subprocess.call([f'{vpp_path}/v++', '--platform', platform, '-c', '-k', kernel, '--temp_dir', f'./{output}.temp', '-o', f'{output}.linked.xo', f'{output}.linked.xpirbc'], env=my_env)
    if ret != 0 :
        print('Error in v++ compilation ')
        return
        
    ret = subprocess.call([f'{vpp_path}/v++', '--platform', platform, '-l', '--temp_dir', f'./{output}.temp_link', '-o', f'{output}.linked.xclbin', f'{output}.linked.xo'], env=my_env)
    if ret != 0 :
        print('Error in v++ linking')
        return

arg_parser = argparse.ArgumentParser(description='SPIRV-to-XCLBIN Converter')
arg_parser.add_argument(                  dest='ifile',   metavar='FILE', type=str, help='Input file')
arg_parser.add_argument('-o', '--output', dest='ofile',   required=True,           help='Output file')
arg_parser.add_argument('-k', '--kernel', dest='kernel',   required=True,           help='Kernel name')
arg_parser.add_argument('-l', '--platform', dest='platform',   required=True,           help='Xilinx platform to target. You can check the available platforms with platoformid -l')
args = arg_parser.parse_args()


generate_xclbin(args.ifile, args.ofile, args.kernel, args.platform)