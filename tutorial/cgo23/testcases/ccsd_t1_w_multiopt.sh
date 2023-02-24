fname="ccsd-t1_11.ta"

sharedlib_ext=".dylib"

$COMET_OPT              \
    --convert-to-loops  \
    --convert-tc-to-ttgt \
    --opt-matmul-tiling \
    --opt-bestperm-ttgt \
    --opt-dense-transpose \
    --opt-multiop-factorize \
    ../benchs/$fname &> ../IRs/$fname-ccsdt1_w_multiopt.mlir

$MLIR_OPT \
    --lower-affine \
    --convert-linalg-to-loops \
    --convert-linalg-to-std \
    --convert-linalg-to-llvm \
    --convert-scf-to-std \
    --convert-std-to-llvm \
    ../IRs/$fname-ccsdt1_w_multiopt.mlir > ../IRs/$fname-ccsdt1_w_multiopt.llvm

$MLIR_CPU_RUNNER ../IRs/$fname-ccsdt1_w_multiopt.llvm \
    -O3 -e main -entry-point-result=void \
    -shared-libs=$COMET_LIB/libcomet_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_runner_utils$sharedlib_ext,$MLIR_LIB/libmlir_c_runner_utils$sharedlib_ext
