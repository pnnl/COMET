#!/bin/bash

# List of testcases to run 
testcases=(
        # "mult_dense_matrix" 
        # "ccsd_t1_21" 
        # "intensli1"
        "spmm_coo"
        "spmm_csr")

output_dir="../outputs/09.20.6/"

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

# Loop over each testcases and run them
for testcase in "${testcases[@]}"
do
    echo "Running the testcases: $testcase"
    
    # Execute the command with the current input
    bash ./"$testcase".sh &> "$output_dir"/"$testcase".out
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Command failed with input: $testcase"
    else
        echo "Command succeeded with input: $testcase"
    fi
done
