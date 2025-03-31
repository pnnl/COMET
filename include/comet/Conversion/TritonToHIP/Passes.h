#ifndef TRITON_TO_HIP_CONVERSION_PASSES
#define TRITON_TO_HIP_CONVERSION_PASSES

#include "comet/Conversion/TritonToHIP/TritonToHIPPass.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/TritonToHIP/Passes.h.inc"
}
}

#endif