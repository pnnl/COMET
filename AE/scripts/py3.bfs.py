import os
import sys
import subprocess
import pandas as pd

################
# Run under AE/
################

EXT = os.environ["EXT"]
DATA_DIR = os.environ["DATA_DIR"]

# LAGraph
LAGRAPH_EXE_DIR = os.environ["LAGRAPH_EXE_DIR"]

# GraphX
COMET_OPT = os.environ["COMET_OPT"]
COMET_OPT_OPTIONS = os.environ["COMET_OPT_OPTIONS"]

# LLVM and MLIR
MLIR_CPU_RUNNER = os.environ["MLIR_CPU_RUNNER"]
SHARED_LIBS = os.environ["SHARED_LIBS"]
MLIR_OPT = os.environ["MLIR_OPT"]
MLIR_OPT_OPTIONS = os.environ["MLIR_OPT_OPTIONS"]


def get_elapsed_time(columns: list) -> float:
    index = 0
    for col in columns:
        if col == "ELAPSED_TIME":
            runtime = float(columns[index + 2])
            break
        index += 1

    return runtime


#######################
# Run GraphX
#######################
def run_GraphX(input_ta: str,
               matrices: list,
               runtimes: list,
               num_rounds: int):

    basename = os.path.basename(input_ta)
    # command1 = F"{COMET_OPT} {COMET_OPT_OPTIONS} {input_ta} &> results/{basename}.llvm"
    # subprocess.run(command1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command2 = F"{MLIR_OPT} {MLIR_OPT_OPTIONS} {input_ta} &> results/{basename}.llvm"
    subprocess.run(command2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command3 = F"{MLIR_CPU_RUNNER} results/{basename}.llvm -O3 -e main -entry-point-result=void -shared-libs={SHARED_LIBS}"
    
    # Get the runtime for every matrix
    for mtx in matrices:
        input_matrix = F"{DATA_DIR}/{mtx}/{mtx}.mtx"
        env_vars = os.environ
        env_vars["SPARSE_FILE_NAME0"] = input_matrix
        env_vars["SPARSE_FILE_NAME1"] = input_matrix
        env_vars["SPARSE_FILE_NAME2"] = input_matrix
    
        print(F"\n#### GraphX: {basename} ####")
    
        total_time = 0
        for r_i in range(num_rounds):
            result = subprocess.run(command3, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            columns = result.stdout.decode("utf-8").split()
            runtime = get_elapsed_time(columns)
            print(F"round: {r_i + 1} runtime: {runtime}")  # test
            total_time += runtime
        avg_time = total_time / num_rounds
        runtimes.append(avg_time)

        print(F"matrix: {mtx} avg_time: {avg_time}")

# End of run_GraphX()


def run_GraphBLAS(exe: str,
                  matrices: list,
                  runtimes: list):

    for mtx in matrices:
        input_matrix = F"{DATA_DIR}/{mtx}/{mtx}.mtx"
        print(F"\n#### GraphBLAS: {exe} ####")

        command = F"{LAGRAPH_EXE_DIR}/{exe} {input_matrix} {input_matrix}"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_out = result.stdout.decode("utf-8")
        print(print_out)
        columns = print_out.split()
        avg_time = float(columns[-3])
        runtimes.append(avg_time)
        print(F"matrix: {mtx} avg_time: {avg_time}")
# End of run_GraphBLAS()


def main():
    if len(sys.argv) != 4:
        print(F"Usage: python3 {sys.argv[0]} <input.ta> <output.csv> <LAGraph_exe>")
        exit(-1)
    input_ta = sys.argv[1]
    output_csv = sys.argv[2]
    LAGraph_exe = sys.argv[3]

    all_matrices = ["bcsstk17",
                    "pdb1HYS",
                    "rma10",
                    "cant",
                    "consph",
                    "shipsec1",
                    "cop20k_A",
                    "scircuit",
                    "com-Orkut",
                    "com-LiveJournal"]
    num_rounds = 4

    # Run GraphX
    runtimes_graphx = []
    run_GraphX(input_ta=input_ta,
               matrices=all_matrices,
               runtimes=runtimes_graphx,
               num_rounds=num_rounds)

    # Test
    # runtimes_graphx = [0.1] * len(all_matrices)
    # End Test

    # Run LAGraph
    runtimes_graphblas = []
    run_GraphBLAS(exe=LAGraph_exe,
                  matrices=all_matrices,
                  runtimes=runtimes_graphblas)

    # Test
    # runtimes_graphblas = [1.0] * len(all_matrices)
    # End test

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)

    columns = {
        'Matrices': all_matrices,
        'LAGraph.BFS': runtimes_graphblas,
        'GraphX.BFS': runtimes_graphx,
    }
    dataFrame = pd.DataFrame(data=columns)
    print('#################')
    print(dataFrame)
    dataFrame.to_csv(F"{output_csv}")


if __name__ == "__main__":
    main()

