#ifndef COMET_BLOCKED_GPU_TO_TRITON_CONVERSION_PASSES
#define COMET_BLOCKED_GPU_TO_TRITON_CONVERSION_PASSES

#include "comet/Conversion/BlockedGpuToTriton/BlockedGpuToTriton.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/BlockedGpuToTriton/Passes.h.inc"
}
}

#endif