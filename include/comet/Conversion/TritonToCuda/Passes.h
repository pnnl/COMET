#ifndef TRITON_TO_CUDA_CONVERSION_PASSES
#define TRITON_TO_CUDA_CONVERSION_PASSES

#include "comet/Conversion/TritonToCuda/TritonToCudaPass.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/TritonToCuda/Passes.h.inc"
}
}

#endif