#!/bin/bash

export SPARSE_FILE_NAME1=first.mtx

#build/bin/comet-opt --convert-ta-to-it --convert-to-loops first.ta &> first.mlir
#build/bin/comet-opt --convert-ta-to-it first.ta &> first.mlir

build/bin/comet-opt --convert-ta-to-it --convert-to-loops --convert-to-llvm first.ta &> first.mlir

llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void \
	-shared-libs=build/lib/libcomet_runner_utils.dylib,llvm/build/lib/libmlir_runner_utils.dylib

# llvm/build/bin/mlir-cpu-runner first.mlir -O3 -e main -entry-point-result=void -shared-libs=build/lib/libcomet_runner_utils.dylib,llvm/build/lib/libmlir_runner_utils.dylib

