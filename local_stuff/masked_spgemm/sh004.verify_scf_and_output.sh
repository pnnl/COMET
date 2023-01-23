if [[ !($# -eq 1) ]]; then
    echo "Usage: bash $0 <input.scf>"
    exit
fi
scf_file=$1
base_name=$(basename "${scf_file}" ".mlir")

#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/rma10/rma10.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/rma10/rma10.mtx
#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/mac_econ_fwd500/mac_econ_fwd500.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/mac_econ_fwd500/mac_econ_fwd500.mtx
#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/pwtk/pwtk.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/pwtk/pwtk.mtx
#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
#export SPARSE_FILE_NAME0=../../integration_test/data/test_rank2.mtx
#export SPARSE_FILE_NAME1=../../integration_test/data/test_rank2.mtx
#export SPARSE_FILE_NAME2=./data/mask.mtx
#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/bcsstk29/bcsstk29.mtx
#export SPARSE_FILE_NAME2=/Users/peng599/pppp/CLion/COMET_masking/local_stuff/masked_spgemm/data/bcsstk29.ones.mtx
#export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/shipsec1/shipsec1.mtx
#export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/shipsec1/shipsec1.mtx
#export SPARSE_FILE_NAME2=/Users/peng599/pppp/CLion/COMET_masking/local_stuff/masked_spgemm/data/shipsec1.ones.mtx
export SPARSE_FILE_NAME0=/Users/peng599/local/react-eval/matrices/cant/cant.mtx
export SPARSE_FILE_NAME1=/Users/peng599/local/react-eval/matrices/cant/cant.mtx
export SPARSE_FILE_NAME2=/Users/peng599/pppp/CLion/COMET_masking/local_stuff/masked_spgemm/data/cant.ones.mtx

# ../../cmake-build-debug/bin/comet-opt --convert-ta-to-it --convert-to-loops mult_spgemm_CSRxCSR_oCSR.mask.ta &> mult_spgemm_CSRxCSR_oCSR.mask.ta.loops.mlir

# ../../cmake-build-debug/bin/comet-opt --convert-scf-to-std --convert-std-to-llvm mult_spgemm_CSRxCSR_oCSR.mask.loops.manual.v0.mlir &> mult_spgemm_CSRxCSR_oCSR.mask.loops.manual.v0.llvm

../../llvm/build/bin/mlir-opt --convert-scf-to-std --convert-std-to-llvm "${scf_file}" &> "${base_name}.llvm"

# echo "${base_name}.llvm.mlir"

../../llvm/build/bin/mlir-cpu-runner "${base_name}.llvm" -O3 -e main -entry-point-result=void -shared-libs=../../cmake-build-debug/lib/libcomet_runner_utils.dylib,../../llvm/build/lib/libmlir_runner_utils.dylib,../../llvm/build/lib/libmlir_c_runner_utils.dylib

