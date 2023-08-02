#!/bin/bash

# 2D
export SPARSE_FILE_NAME0=integration_test/data/test_rank2.mtx
export SPARSE_FILE_NAME1=integration_test/data/test_rank2.mtx
#export SPARSE_FILE_NAME=integration_test/data/test_rank2.mtx

#build/bin/comet-opt --convert-ta-to-it --convert-to-loops first.ta &> first.mlir
#build/bin/comet-opt --convert-ta-to-it first.ta &> first.mlir

build/bin/comet-opt --opt-comp-workspace --convert-ta-to-it --convert-to-loops --convert-to-llvm first.ta &> first.mlir

llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
	-shared-libs=build/lib/libcomet_runner_utils.dylib,llvm/build/lib/libmlir_runner_utils.dylib

# llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void -shared-libs=build/lib/libcomet_runner_utils.dylib,llvm/build/lib/libmlir_runner_utils.dylib

