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
               input_scf: str,
               matrices: list,
               runtimes: list,
               num_rounds: int):

    ta_file = os.path.basename(input_ta)
    command1 = F"{COMET_OPT} {COMET_OPT_OPTIONS} {input_ta}"
    result = subprocess.run(command1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(F"results/{ta_file}.llvm", "w") as fout:
        fout.write(result.stdout.decode("utf-8"))
    scf_file = os.path.basename(input_scf)
    command2 = F"{MLIR_OPT} {MLIR_OPT_OPTIONS} {input_scf}"
    result = subprocess.run(command2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(F"results/{scf_file}.llvm", "w") as fout:
        fout.write(result.stdout.decode("utf-8"))
    command3 = F"{MLIR_CPU_RUNNER} results/{ta_file}.llvm -O3 -e main -entry-point-result=void -shared-libs={SHARED_LIBS}"
    command4 = F"{MLIR_CPU_RUNNER} results/{scf_file}.llvm -O3 -e main -entry-point-result=void -shared-libs={SHARED_LIBS}"

    # Get the runtime for every matrix
    for mtx in matrices:
        if mtx not in ["com-Orkut", "com-LiveJournal"]:
            cmd = command3
        else:
            cmd = command4
        input_matrix = F"{DATA_DIR}/{mtx}/{mtx}.mtx"
        env_vars = os.environ
        env_vars["SPARSE_FILE_NAME0"] = input_matrix
        env_vars["SPARSE_FILE_NAME1"] = input_matrix
        env_vars["SPARSE_FILE_NAME2"] = input_matrix
    
        print(F"\n#### GraphX: {os.path.splitext(ta_file)[0]} ####")
    
        total_time = 0
        for r_i in range(num_rounds):
            result = subprocess.run(cmd, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print_out = result.stdout.decode("utf-8")
        print(print_out)
        columns = print_out.split()
        avg_time = float(columns[-3])
        runtimes.append(avg_time)
        print(F"matrix: {mtx} avg_time: {avg_time}")
# End of run_GraphBLAS()


def main():
    if len(sys.argv) != 5:
        print(F"Usage: python3 {sys.argv[0]} <input.ta> <input.scf> <output.csv> <LAGraph_exe>")
        exit(-1)
    input_ta = sys.argv[1]
    input_scf = sys.argv[2]
    output_csv = sys.argv[3]
    LAGraph_exe = sys.argv[4]

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

    # Test
    # all_matrices = ["bcsstk17",
    #                 "com-LiveJournal",
    #                 "com-Orkut"]
    # num_rounds = 1
    # End Test

    # Run GraphX
    runtimes_graphx = []
    run_GraphX(input_ta=input_ta,
               input_scf=input_scf,
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
        'LAGraph.SpGEMM+Masking': runtimes_graphblas,
        'GraphX.SpGEMM+Masking': runtimes_graphx,
    }
    dataFrame = pd.DataFrame(data=columns)
    print('#################')
    print(dataFrame)
    dataFrame.to_csv(F"{output_csv}")


if __name__ == "__main__":
    main()

