fname="ccsd_t1_21.ta"

sharedlib_ext=".dylib"

$COMET_BIN_DIR/comet-opt   \
    --convert-ta-to-it \
    --convert-to-loops  \
    --convert-to-llvm  \
    ../$fname &> ../IRs/$fname-no-opt.llvm

echo "Execution Time for COMET *WITHOUT* Optimization:"
$MLIR_BIN_DIR/mlir-cpu-runner ../IRs/$fname-no-opt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB_DIR/libcomet_runner_utils$sharedlib_ext


$COMET_BIN_DIR/comet-opt   \
    -opt-bestperm-ttgt   \
    -opt-matmul-tiling   \
    -opt-matmul-mkernel  \
    -opt-dense-transpose \
    --convert-tc-to-ttgt \
    --convert-to-llvm    \
    ../$fname &> ../IRs/$fname-opt.llvm

echo "Execution Time for COMET *WITH* Optimization:"
$MLIR_BIN_DIR/mlir-cpu-runner ../IRs/$fname-opt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB_DIR/libcomet_runner_utils$sharedlib_ext