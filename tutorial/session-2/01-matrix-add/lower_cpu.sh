COMET_BIN=../../../build/bin/comet-opt
${COMET_BIN} --target=CPU --emit-llvm --mlir-print-ir-after-all $1 &> passes_cpu.txt

