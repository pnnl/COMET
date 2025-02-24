import subprocess
import sys
lines = None
llvm_spirv_path = "/Users/thom895/local/TwoTruths/COMET/llvm-7/build/bin"
llvm_hls_path = "/share/apps/vitis_2021.1/Vitis_HLS/2021.1/lnx64/tools/clang-3.9-csynth/bin/"
vpp_path = "/share/apps/vitis_2021.1/Vitis/2021.1/bin/"
libspir_hls_path = "/share/apps/vitis_2021.1/Vitis_HLS/2021.1/lnx64/lib/libspir-39-hls.bc"
platform = "xilinx_vck5000_gen3x16_xdma_1_202120_1"
kernel = 'main_kernel'
p = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', "-to-text", "-o", f"{sys.argv[1]}.spt", f"{sys.argv[1]}"])
with open(f'{sys.argv[1]}.spt', 'r') as f:
    vars = []
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "BuiltIn 26" in line or "BuiltIn 27" in line or "BuiltIn 28" in line:
            if "Decorate " in line:
                vars.append(line.split()[2])
        elif "4 Variable 1" in line and line.strip().endswith("1"):
            lines[i] = line.strip()[:-1]+"0\n" 

with open(f'{sys.argv[1]}.spt', 'w') as f:
    f.writelines(lines)

p = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', "-to-binary", "-o", f"{sys.argv[1]}.spv", f"{sys.argv[1]}.spt"])
p = subprocess.call([f'{llvm_spirv_path}/llvm-spirv', "-r", "-o", f"{sys.argv[1]}.bc", f"{sys.argv[1]}.spv"])
p = subprocess.call([f'{llvm_spirv_path}/llvm-dis', "-o", f"{sys.argv[1]}.ll", f"{sys.argv[1]}.bc"])
p = subprocess.call([f'{llvm_hls_path}/llvm-as', '--only-needed', "-o", f"{sys.argv[1]}.xpirbc", f"{sys.argv[1]}.ll"])
p = subprocess.call([f'{llvm_hls_path}/llvm-link', "-o", f"{sys.argv[1]}.linked.xpirbc", f"{sys.argv[1]}.xpirbc", f'{libspir_hls_path}'])
p = subprocess.call([f'{vpp_path}/v++', '--platform', f'{platform}', '-c', '-k', f'{kernel}', '--temp_dir', './temp', '-o', f"{sys.argv[1]}.linked.xo", f"{sys.argv[1]}.linked.xpirbc"])
p = subprocess.call([f'{vpp_path}/v++', '--platform', f'{platform}', '-l', '--temp_dir', './temp_link', '-o', f"{sys.argv[1]}.linked.xclbin", f"{sys.argv[1]}.linked.xo"])