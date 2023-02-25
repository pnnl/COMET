fname="gnn.ta"
export SPARSE_FILE_NAME0=../inputs/pdb1HYS.mtx

sharedlib_ext=".dylib"

$COMET_OPT              \
    --convert-ta-to-it  \
    --opt-fusion \
    --convert-to-loops  \
    ../benchs/$fname &> ../IRs/$fname-w-fusion.mlir

$MLIR_OPT \
    --lower-affine \
    --convert-linalg-to-loops \
    --convert-linalg-to-std \
    --convert-linalg-to-llvm \
    --convert-scf-to-std \
    --convert-std-to-llvm \
    ../IRs/$fname-w-fusion.mlir > ../IRs/$fname-w-fusion.llvm

$MLIR_CPU_RUNNER ../IRs/$fname-w-fusion.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_c_runner_utils$sharedlib_ext

