#ifndef COMET_GPU_TO_BLOCKED_GPU_CONVERSION_PASSES
#define COMET_GPU_TO_BLOCKED_GPU_CONVERSION_PASSES

#include "comet/Conversion/GpuToBlockedGpu/GpuToBlockedGpu.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/GpuToBlockedGpu/Passes.h.inc"
}
}

#endif