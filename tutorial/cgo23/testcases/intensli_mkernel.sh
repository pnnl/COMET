fname="intensli1.ta"

sharedlib_ext=".dylib"

$COMET_OPT              \
    --convert-to-loops  \
    --convert-tc-to-ttgt \
    --opt-matmul-tiling \
    --opt-matmul-mkernel \
    ../benchs/$fname &> ../IRs/$fname-mkernel.mlir

$MLIR_OPT \
    --lower-affine \
    --convert-linalg-to-loops \
    --convert-linalg-to-std \
    --convert-linalg-to-llvm \
    --convert-scf-to-std \
    --convert-std-to-llvm \
    ../IRs/$fname-mkernel.mlir > ../IRs/$fname-mkernel.llvm

$MLIR_CPU_RUNNER ../IRs/$fname-mkernel.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_c_runner_utils$sharedlib_ext

