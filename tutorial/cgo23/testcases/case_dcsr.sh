fname="mult_spmm_DCSRxDense.ta"
export SPARSE_FILE_NAME0=../inputs/pwtk.mtx

$COMET_OPT              \
    --convert-ta-to-it  \
    --convert-to-loops  \
    ../benchs/$fname &> ../IRs/$fname-loop.mlir

$MLIR_OPT \
    --lower-affine \
    --convert-linalg-to-loops \
    --convert-linalg-to-std \
    --convert-linalg-to-llvm \
    --convert-scf-to-std \
    --convert-std-to-llvm \
    ../IRs/$fname-loop.mlir > ../IRs/$fname-loop.llvm

$MLIR_CPU_RUNNER ../IRs/$fname-loop.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB/libcomet_runner_utils.so,$MLIR_LIB/libmlir_runner_utils.so,$MLIR_LIB/libmlir_c_runner_utils.so
