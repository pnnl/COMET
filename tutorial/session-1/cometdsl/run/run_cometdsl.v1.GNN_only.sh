#!/bin/bash

set -euo pipefail

source ../../../cometenv.sh

# List of testcases to run 
dense_testcases=(
        "mult_dense_matrix"
        "ccsd_t1_21_opt1"
        "ccsd_t1_21_opt2"
        "ccsd_t1_21_opt3"
        "intensli1"
        )

sparse_testcases=(
#        "spmm_coo"
#        "spmm_csr"
#        "spgemm_csr"
#        "semiring_pluspair"
        "gnn"
#        "tc_SandiaLL"
#        "tc_SandiaLL_wSemiring_wMasking"
        )

sparse_inputs=(
        "bcsstk17"
        "cant"
        "consph"
        "cop20k_A"
        "pdb1HYS"
        "rma10"
        "scircuit"
        "shipsec1"
        # "bcsstk17.unweighted"
        # "bcsstk29.unweighted"
        # "cant.unweighted"
        # "consph.unweighted"
        # "cop20k_A.unweighted"
        )

function set_output_dir () {
    output_dir=$1
    if [ ! -d "$output_dir" ]; then
        echo "Directory $output_dir does not exist. Creating it now."
        mkdir -p "$output_dir"
        if [ $? -eq 0 ]; then
            echo "Directory $output_dir created successfully."
        else
            echo "Failed to create directory $output_dir."
            exit 1
        fi
    else
        echo "Directory $output_dir already exists."
    fi
}

# Clear old results
rm -rf ../results

## Loop over dense testcases and run them
#for dense_testcase in "${dense_testcases[@]}"
#do
#    echo "Running the dense testcases: $dense_testcase"
##    set_output_dir "../results/${dense_testcase}"
#    set_output_dir "../results/dense"
#    bash ./"$dense_testcase".sh &> "${output_dir}/${dense_testcase}.out"
#    avg_exe_time="$(python ../py_scripts/py00.get_avg_time.py ${output_dir}/${dense_testcase}.out)"
#    echo "${dense_testcase},${avg_exe_time}" >> "${output_dir}/dense-all-collects.out"
#done

# Loop over sparse testcases and run them
for sparse_testcase in "${sparse_testcases[@]}"
do
    echo "Running the sparse testcases: $sparse_testcase"
    set_output_dir "../results/${sparse_testcase}"
    for sinput in "${sparse_inputs[@]}"
    do
        export SPARSE_FILE_NAME1="../../../data/${sinput}/${sinput}.mtx"
        export SPARSE_FILE_NAME0="../../../data/${sinput}/${sinput}.mtx"
        echo $SPARSE_FILE_NAME0
        # Execute the command with the current input
        bash ./"$sparse_testcase".sh &> "${output_dir}/${sparse_testcase}-${sinput}.out"
        avg_no_opti_exe_time="$(python ../py_scripts/py01.get_no_opti_avg_time.py ${output_dir}/${sparse_testcase}-${sinput}.out)"
        avg_exe_time="$(python ../py_scripts/py00.get_avg_time.py ${output_dir}/${sparse_testcase}-${sinput}.out)"
        echo "${sinput},${avg_no_opti_exe_time},${avg_exe_time}" >> "${output_dir}/${sparse_testcase}-all-collects.out"
    done
    
done
