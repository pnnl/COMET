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


#
# Get Elapsed Time (runtime) from GraphX's print-out
def get_elapsed_time(columns: list) -> float:
    index = 0
    for col in columns:
        if col == "ELAPSED_TIME":
            runtime = float(columns[index + 2])
            break
        index += 1

    return runtime


#
# Run a given GraphX code
def run_GraphX_code(input_ta: str,
                    input_scf: str,
                    matrices: list,
                    runtimes: list,
                    num_rounds: int):
    ta_file = os.path.basename(input_ta)
    command1 = F"{COMET_OPT} {COMET_OPT_OPTIONS} {input_ta} &> results/{ta_file}.llvm"
    subprocess.run(command1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    scf_file = os.path.basename(input_scf)
    command2 = F"{MLIR_OPT} {MLIR_OPT_OPTIONS} {input_scf} &> results/{scf_file}.llvm"
    subprocess.run(command2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            result = subprocess.run(cmd, env=env_vars, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            columns = result.stdout.decode("utf-8").split()
            runtime = get_elapsed_time(columns)
            print(F"round: {r_i + 1} runtime: {runtime}")  # test
            total_time += runtime
        avg_time = total_time / num_rounds
        runtimes.append(avg_time)

        print(F"matrix: {mtx} avg_time: {avg_time}")


#######################
# Run GraphX
#######################
def run_GraphX(TC_Burkhardt_ta: str,
               TC_Burkhardt_scf: str,
               TC_Cohen_ta: str,
               TC_Cohen_scf: str,
               TC_Sandia_LL_ta: str,
               TC_Sandia_LL_scf: str,
               TC_Sandia_UU_ta: str,
               TC_Sandia_UU_scf: str,
               matrices: list,
               runtimes: list,
               num_rounds: int):
    # Burkhardt
    rt_list = []
    run_GraphX_code(input_ta=TC_Burkhardt_ta,
                    input_scf=TC_Burkhardt_scf,
                    matrices=matrices,
                    runtimes=rt_list,
                    num_rounds=num_rounds)
    runtimes.append(rt_list)

    # Cohen
    rt_list = []
    run_GraphX_code(input_ta=TC_Cohen_ta,
                    input_scf=TC_Cohen_scf,
                    matrices=matrices,
                    runtimes=rt_list,
                    num_rounds=num_rounds)
    runtimes.append(rt_list)

    # Sandia_LL
    rt_list = []
    run_GraphX_code(input_ta=TC_Sandia_LL_ta,
                    input_scf=TC_Sandia_LL_scf,
                    matrices=matrices,
                    runtimes=rt_list,
                    num_rounds=num_rounds)
    runtimes.append(rt_list)

    #Sandia_UU
    rt_list = []
    run_GraphX_code(input_ta=TC_Sandia_UU_ta,
                    input_scf=TC_Sandia_UU_scf,
                    matrices=matrices,
                    runtimes=rt_list,
                    num_rounds=num_rounds)
    runtimes.append(rt_list)
# End of run_GraphX()


#######################
# Run LAGraph
#######################
def run_GraphBLAS(exe: str,
                  matrices: list,
                  runtimes: list):

    burkhardt_rt = []
    cohen_rt = []
    sandia_ll_rt = []
    sandia_uu_rt = []

    for mtx in matrices:
        input_matrix = F"{DATA_DIR}/{mtx}/{mtx}.mtx"
        print(F"\n#### GraphBLAS: {exe} ####")

        command = F"{LAGRAPH_EXE_DIR}/{exe} {input_matrix} {input_matrix}"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_out = result.stdout.decode("utf-8")
        print(print_out)
        rows = print_out.split("\n")
        result_lines = []
        count = 0
        for line in rows:
            if line.startswith("nthreads:   1 time:"):
                count += 1
                if count == 2:
                    columns = line.split()
                    time = float(columns[3])
                    burkhardt_rt.append(time)
                elif count == 3:
                    columns = line.split()
                    time = float(columns[3])
                    cohen_rt.append(time)
                elif count == 4:
                    columns = line.split()
                    time = float(columns[3])
                    sandia_ll_rt.append(time)
                elif count == 5:
                    columns = line.split()
                    time = float(columns[3])
                    sandia_uu_rt.append(time)

    runtimes.append(burkhardt_rt)
    runtimes.append(cohen_rt)
    runtimes.append(sandia_ll_rt)
    runtimes.append(sandia_uu_rt)
# End of run_GraphBLAS()


def main():
    if len(sys.argv) != 11:
        print(F"Usage: python3 {sys.argv[0]} "
              F"<GraphX_TC_Burkhardt.ta> <GraphX_TC_Burkhardt.scf> "
              F"<GraphX_TC_Cohen.ta> <GraphX_TC_Cohen.scf>"
              F"<GraphX_TC_Sandia_LL.ta> <GraphX_TC_Sandia_LL.scf>"
              F"<GraphX_TC_Sandia_UU.ta> <GraphX_TC_Sandia_UU.scf>"
              F"<output.csv> <LAGraph_exe>")
        exit(-1)
    GraphX_TC_Burkhardt_ta = sys.argv[1]
    GraphX_TC_Burkhardt_scf = sys.argv[2]
    GraphX_TC_Cohen_ta = sys.argv[3]
    GraphX_TC_Cohen_scf = sys.argv[4]
    GraphX_TC_Sandia_LL_ta = sys.argv[5]
    GraphX_TC_Sandia_LL_scf = sys.argv[6]
    GraphX_TC_Sandia_UU_ta = sys.argv[7]
    GraphX_TC_Sandia_UU_scf = sys.argv[8]
    output_csv = sys.argv[9]
    LAGraph_exe = sys.argv[10]

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
    run_GraphX(TC_Burkhardt_ta=GraphX_TC_Burkhardt_ta,
               TC_Burkhardt_scf=GraphX_TC_Burkhardt_scf,
               TC_Cohen_ta=GraphX_TC_Cohen_ta,
               TC_Cohen_scf=GraphX_TC_Cohen_scf,
               TC_Sandia_LL_ta=GraphX_TC_Sandia_LL_ta,
               TC_Sandia_LL_scf=GraphX_TC_Sandia_LL_scf,
               TC_Sandia_UU_ta=GraphX_TC_Sandia_UU_ta,
               TC_Sandia_UU_scf=GraphX_TC_Sandia_UU_scf,
               matrices=all_matrices,
               runtimes=runtimes_graphx,
               num_rounds=num_rounds)

    # Test
    # for _ in range(4):
    #     runtimes_graphx.append([1.0] * len(all_matrices))
    # End Test

    # Run LAGraph
    runtimes_graphblas = []
    run_GraphBLAS(exe=LAGraph_exe,
                  matrices=all_matrices,
                  runtimes=runtimes_graphblas)

    # Test
    # for _ in range(4):
    #     runtimes_graphblas.append([1.0] * len(all_matrices))
    # End test

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)

    columns = {
        'Matrices': all_matrices,
        'LAGraph.Burkhardt': runtimes_graphblas[0],
        'LAGraph.Cohen': runtimes_graphblas[1],
        'LAGraph.Sandia_LL': runtimes_graphblas[2],
        'LAGraph.Sandia_UU': runtimes_graphblas[3],
        'GraphX.Burkhardt': runtimes_graphx[0],
        'GraphX.Cohen': runtimes_graphx[1],
        'GraphX.Sandia_LL': runtimes_graphx[2],
        'GraphX.Sandia_UU': runtimes_graphx[3],
    }
    dataFrame = pd.DataFrame(data=columns)
    print('#################')
    print(dataFrame)
    dataFrame.to_csv(F"{output_csv}")


if __name__ == "__main__":
    main()

