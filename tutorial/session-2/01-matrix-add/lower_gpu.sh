COMET_BIN=../../../build/bin/comet-opt
${COMET_BIN} --target=GPU --emit-llvm --mlir-print-ir-after-all matadd.ta &> passes_gpu.txt

