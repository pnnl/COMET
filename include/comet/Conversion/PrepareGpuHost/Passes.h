#ifndef PREPARE_GPU_HOST_PASSES
#define PREPARE_GPU_HOST_PASSES

#include "comet/Conversion/PrepareGpuHost/PrepareGpuHostPass.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/PrepareGpuHost/Passes.h.inc"
}
}

#endif