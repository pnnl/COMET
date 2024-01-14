#!/bin/bash

export LD_LIBRARY_PATH="/home/patrick/Work/PNNL/COMET/install/lib:/home/patrick/Work/PNNL/COMET/build/lib"

export SPARSE_FILE_NAME0=square.mtx
#export SPARSE_FILE_NAME0=integration_test/data/test_rank2.mtx

#build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm first.ta &> first.mlir
#llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
#    -shared-libs=build/lib/libcomet_runner_utils.so

echo "ELL"
build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm ell.mlir &> first.mlir
#llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
#    -shared-libs=build/lib/libcomet_runner_utils.so
llvm/build/bin/mlir-translate --mlir-to-llvmir first.mlir &> /tmp/first.ll

#llvm/build/bin/clang /tmp/first.ll -o /tmp/first -Lbuild/lib -lcomet_runner_utils -march=native -O2 -ftree-vectorize
#/tmp/first

llvm/build/bin/clang /tmp/first.ll -o /tmp/first -Lbuild/lib -lcomet_runner_utils -march=native -O2 -ftree-vectorize -g
perf stat /tmp/first

echo ""

echo "CSR"
build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm csr.mlir &> first.mlir
#llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
#    -shared-libs=build/lib/libcomet_runner_utils.so
llvm/build/bin/mlir-translate --mlir-to-llvmir first.mlir &> /tmp/first.ll

#llvm/build/bin/clang /tmp/first.ll -o /tmp/first -Lbuild/lib -lcomet_runner_utils -march=native -O2 -ftree-vectorize
#/tmp/first

llvm/build/bin/clang /tmp/first.ll -o /tmp/first -Lbuild/lib -lcomet_runner_utils -march=native -O2 -ftree-vectorize -g
perf stat /tmp/first
  
