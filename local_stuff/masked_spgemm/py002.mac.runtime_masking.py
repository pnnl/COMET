import os
import sys
import subprocess
import pandas as pd

if len(sys.argv) != 5:
    print(F"{sys.argv[0]} <project_root> <input.scf> <matrix> <num of rounds>")
    exit(-1)

project_root = sys.argv[1]
input_ta = sys.argv[2]
input_matrix = sys.argv[3]
num_rounds = int(sys.argv[4])

comet_opt = F"{project_root}/cmake-build-debug/bin/comet-opt"
mlir_opt = F"{project_root}/llvm/build/bin/mlir-opt"
mlir_cpu_runner = F"{project_root}/llvm/build/bin/mlir-cpu-runner"
shared_libs = F"{project_root}/cmake-build-debug/lib/libcomet_runner_utils.so," \
              F"{project_root}/llvm/build/lib/libmlir_runner_utils.so," \
              F"{project_root}/llvm/build/lib/libmlir_c_runner_utils.so"

comet_opt_options = "--convert-ta-to-it --convert-to-loops"
mlir_opt_options = "--lower-affine --convert-linalg-to-loops --convert-linalg-to-std " \
                   "--convert-linalg-to-llvm --convert-scf-to-std --convert-std-to-llvm"

basename = os.path.basename(input_ta)
# command1 = F"{comet_opt} {comet_opt_options} {input_ta} &> {basename}.mlir"
command2 = F"{mlir_opt} {mlir_opt_options} {input_ta} &> {basename}.llvm"
command3 = F"{mlir_cpu_runner} {basename}.llvm -O3 -e main -entry-point-result=void -shared-libs={shared_libs}"

env_vars = os.environ
env_vars["SPARSE_FILE_NAME0"] = input_matrix
env_vars["SPARSE_FILE_NAME1"] = input_matrix
env_vars["SPARSE_FILE_NAME2"] = input_matrix

matrix_name = os.path.basename(input_matrix)

# subprocess.run(command1, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(command2, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

total_time = 0
for _ in range(num_rounds):
    result = subprocess.run(command3, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    columns = result.stdout.decode("utf-8").split()
    index = 0
    for col in columns:
        if col == "ELAPSED_TIME":
            runtime = float(columns[index + 2])
            break
        index += 1
    # print(F"runtime: {runtime}")  # test
    total_time += runtime


# print(F"command:\t{sys.argv[0]}\tinput:\t{os.path.basename(input_ta)}\tmatrix:\t{os.path.basename(input_matrix)}")

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
columns = {
    'command': [sys.argv[0]],
    'code_file': [os.path.basename(input_ta)]
}
print(pd.DataFrame(data=columns))
print('#### --------- ####\n')

columns = {
    'matrix': [matrix_name],
    'rounds': [num_rounds],
    'total_time': [total_time],
    'avg_time': [total_time / num_rounds]
}
print(pd.DataFrame(data=columns))
print('#### --------- ####\n')

# print(F"command:\t{sys.argv[0]}\tinput:\t{os.path.basename(input_ta)}\tmatrix:\t{os.path.basename(input_matrix)}\t"
#       F"rounds:\t{num_rounds}\ttotal_time:\t{total_time}\tavg_time:\t{total_time / num_rounds}")
