#
# Test if this machine is running macOS
uname -s | grep -q Darwin
if [ $? -eq 0 ]; then
  # macOS
  export EXT="dylib"
else
  # Not macOS, then Linux
  export EXT="so"
  ulimit -s unlimited  # Set stack size as unlimited
fi


################################################################
# All relative paths are referred to AE/, NOT AE/scripts/, and #
# all command should run under AE/.                            #
################################################################

#
# To include the BLIS lib path into LD_LIBRARY_PATH
blis_path="$(pwd)/../install/lib"
if [[ "${LD_LIBRARY_PATH}" != *"${blis_path}"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${blis_path}"
fi

#
# Data
export DATA_DIR="./data"

#
# LAGraph
# The LAGraph executables are under this directory, such as mxm_serial_demo, tc_demo, and bfs_demo.
export LAGRAPH_EXE_DIR="LAGraph/build/src/benchmark"

#
# GraphX
export GRAPHX_BUILD_DIR="../build"
export COMET_OPT="${GRAPHX_BUILD_DIR}/bin/comet-opt"
export COMET_OPT_OPTIONS="--opt-comp-workspace \
--convert-ta-to-it \
--convert-to-loops \
--convert-to-llvm"

#
# LLVM and MLIR
export LLVM_BUILD_DIR="../llvm/build"
export MLIR_CPU_RUNNER="${LLVM_BUILD_DIR}/bin/mlir-cpu-runner"
export SHARED_LIBS="${GRAPHX_BUILD_DIR}/lib/libcomet_runner_utils.${EXT},\
${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.${EXT},\
${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.${EXT}"

export MLIR_OPT="${LLVM_BUILD_DIR}/bin/mlir-opt"
export MLIR_OPT_OPTIONS="-lower-affine \
-memref-expand \
-convert-scf-to-cf \
-convert-cf-to-llvm \
-convert-vector-to-llvm \
-finalize-memref-to-llvm \
-convert-func-to-llvm \
-reconcile-unrealized-casts"

