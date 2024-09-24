COMET_BIN=../../../build/bin/comet-opt
${COMET_BIN} --target=CPU --emit-llvm --mlir-print-ir-after-all matadd.ta &> passes_cpu.txt

