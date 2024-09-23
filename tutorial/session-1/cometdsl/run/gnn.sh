fname="gnn.ta"

sharedlib_ext=".dylib"
# export SPARSE_FILE_NAME0=../../../data/shipsec1.mtx

# Number of iterations
iterations=50


################## Non Optimized ##################
# Variable to store total time
total_time_noopt=0.0

$COMET_BIN_DIR/comet-opt   \
    --convert-ta-to-it     \
    --convert-to-loops     \
    --convert-to-llvm      \
    ../benchs/$fname &> ../IRs/$fname-non-opt.llvm

command_to_run="$MLIR_BIN_DIR/mlir-cpu-runner ../IRs/$fname-non-opt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB_DIR/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB_DIR/libomp$sharedlib_ext"

for i in $(seq 1 $iterations); do
    # Capture the output of the command
    output=$( $command_to_run )

    # Extract the numeric part from the output using grep (for floats and integers)
    numeric_output=$(echo $output | grep -o -E '[0-9]+([.][0-9]+)?')
    

    # Check if the output is a valid number
    if [[ $numeric_output =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_time_noopt=$(echo "$total_time_noopt + $numeric_output" | bc -l)
        echo "Iteration $i: $numeric_output"
    else
        echo "Iteration $i: No valid numeric value found in output: $numeric_output"
    fi

done


################## Optimized ##################
# Variable to store total time
total_time_opt=0.0

$COMET_BIN_DIR/comet-opt   \
    --convert-ta-to-it     \
    --opt-fusion           \
    --convert-to-loops     \
    --convert-to-llvm       \
    ../benchs/$fname &> ../IRs/$fname-opt.llvm

command_to_run="$MLIR_BIN_DIR/mlir-cpu-runner ../IRs/$fname-opt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB_DIR/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB_DIR/libomp$sharedlib_ext"

for i in $(seq 1 $iterations); do
    # Capture the output of the command
    output=$( $command_to_run )

    # Extract the numeric part from the output using grep (for floats and integers)
    numeric_output=$(echo $output | grep -o -E '[0-9]+([.][0-9]+)?')
    

    # Check if the output is a valid number
    if [[ $numeric_output =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_time_opt=$(echo "$total_time_opt + $numeric_output" | bc -l)
        echo "Iteration $i: $numeric_output"
    else
        echo "Iteration $i: No valid numeric value found in output: $numeric_output"
    fi

done

# # Calculate the average value
average_value=$(echo "$total_time_noopt / $iterations" | bc -l)
echo "Average Execution Time for COMET *WITHOUT* Optimization: $average_value"

# # Calculate the average value
average_value=$(echo "$total_time_opt / $iterations" | bc -l)
echo "Average Execution Time for COMET *WITH* Optimization: $average_value"
