if [[ !($# -eq 1) ]]; then
    echo "Usage: bash $0 <input.ta>"
    exit
fi
ta_file=$1
base_name=$(basename "${ta_file}" ".ta")

#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
export SPARSE_FILE_NAME0=../../integration_test/data/test_rank2.mtx
export SPARSE_FILE_NAME1=../../integration_test/data/test_rank2.mtx
export SPARSE_FILE_NAME2=./data/mask.mtx

../../cmake-build-debug/bin/comet-opt --opt-comp-workspace --convert-ta-to-it --convert-to-loops "${ta_file}" &> "${base_name}.mlir"
#../../cmake-build-debug/bin/comet-opt --convert-ta-to-it --convert-to-loops "${ta_file}" &> "${base_name}.mlir"

../../llvm/build/bin/mlir-opt --lower-affine --convert-linalg-to-loops --convert-linalg-to-std --convert-scf-to-std \
    --convert-linalg-to-llvm  --convert-std-to-llvm "${base_name}.mlir" &> "${base_name}.llvm"

../../llvm/build/bin/mlir-cpu-runner "${base_name}.llvm" -O3 -e main -entry-point-result=void -shared-libs=../../cmake-build-debug/lib/libcomet_runner_utils.dylib,../../llvm/build/lib/libmlir_runner_utils.dylib,../../llvm/build/lib/libmlir_c_runner_utils.dylib

