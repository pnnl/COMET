fname="tc_SandiaLL.ta"

# Detect the operating system
get_shared_lib_extension() {
    case "$(uname -s)" in
        Linux*)
            echo ".so"
            ;;
        Darwin*)
            echo ".dylib"
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            echo ".dll"
            ;;
        *)
            echo "Unknown OS"
            ;;
    esac
}

# Call the function and store the result in a variable
sharedlib_ext=$(get_shared_lib_extension)

# export SPARSE_FILE_NAME0=../../../data/shipsec1.mtx
# export SPARSE_FILE_NAME1=../../../data/shipsec1.mtx

# Number of iterations
iterations=10

# Variable to store total time
total_time=0.0


$COMET_BIN_DIR/comet-opt   \
    --opt-comp-workspace \
    --convert-ta-to-it \
    --convert-to-loops \
    --convert-to-llvm \
    ../benchs/$fname &> ../IRs/$fname-opt.llvm

command_to_run="$MLIR_BIN_DIR/mlir-cpu-runner ../IRs/$fname-opt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB_DIR/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB_DIR/libomp$sharedlib_ext,${MLIR_LIB_DIR}/libmlir_c_runner_utils${sharedlib_ext}"

for i in $(seq 1 $iterations); do
    # Capture the output of the command
    output=$( $command_to_run )

    # Extract the numeric part from the output using grep (for floats and integers)
    numeric_output=$(echo $output | grep -o -E '[0-9]+([.][0-9]+)?$')
    

    # Check if the output is a valid number
    if [[ $numeric_output =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_time=$(echo "$total_time + $numeric_output" | bc -l)
        echo "Iteration $i: $numeric_output"
    else
        echo "Iteration $i: No valid numeric value found in output: $numeric_output"
    fi

done

# # Calculate the average value
average_value=$(echo "$total_time / $iterations" | bc -l)
echo "Average Execution Time for COMET *WITH* Optimization: $average_value"
