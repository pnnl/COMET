#!/bin/bash

# List of testcases to run 
dense_testcases=(
        #"mult_dense_matrix" 
        "ccsd_t1_21_opt1" 
        "ccsd_t1_21_opt2" 
        "ccsd_t1_21_opt3" 
        # "intensli1"
        )

sparse_testcases=(
        # "spmm_COO"
        # "spmm_CSR"
        # "spgemm_CSR"
        # "semiring_PlusPair"
        # "gnn"
        # "tc_SandiaLL"
        # "tc_SandiaLL_wSemiring_wMasking"
        )

output_dir="../results/ccsd-t1/"

sparse_inputs=(
        # "bcsstk17"
        # "cant"
        # "consph"
        # "cop20k_A"
        # "pdb1HYS"
        # "rma10"
        # "scircuit"
        # "shipsec1"
        # "bcsstk17.unweighted"
        # "bcsstk29.unweighted"
        # "cant.unweighted"
        # "consph.unweighted"
        # "cop20k_A.unweighted"
        )

# Check if the directory exists
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

# Loop over dense testcases and run them
for dense_testcase in "${dense_testcases[@]}"
do
    echo "Running the sparse testcases: $dense_testcase"
    bash ./"$dense_testcase".sh &> "$output_dir"/"$dense_testcase".out
done

# Loop over sparse testcases and run them
for sparse_testcase in "${sparse_testcases[@]}"
do
    echo "Running the sparse testcases: $sparse_testcase"
    
    for sinput in "${sparse_inputs[@]}"
    do
        export SPARSE_FILE_NAME0=/Users/kest268/projects/COMET/COMET/tutorial/data/$sinput".mtx"
        export SPARSE_FILE_NAME1=/Users/kest268/projects/COMET/COMET/tutorial/data/$sinput".mtx"
        echo $SPARSE_FILE_NAME0
        # Execute the command with the current input
        bash ./"$sparse_testcase".sh &> "$output_dir"/"$sparse_testcase"-"$sinput".out
    done
    
done
