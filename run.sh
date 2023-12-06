#!/bin/bash

export LD_LIBRARY_PATH="/home/patrick/Work/PNNL/COMET/install/lib"

export SPARSE_FILE_NAME0=first.mtx
#build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm first.ta &> first.mlir
build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm bcsr.mlir &> first.mlir

#export SPARSE_FILE_NAME0=integration_test/data/test_rank2.mtx
#export SPARSE_FILE_NAME1=integration_test/data/test_rank2_transpose.mtx
#build/bin/comet-opt --opt-comp-workspace --convert-ta-to-it --convert-to-loops --convert-to-llvm first.ta &> first.mlir

llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
    -shared-libs=build/lib/libcomet_runner_utils.so
    
