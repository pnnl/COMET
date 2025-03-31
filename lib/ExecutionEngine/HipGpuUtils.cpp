#include "hip/hip_runtime.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t res = call;                                                     \
    if (res != hipSuccess) {                                                   \
      fprintf(stderr, "hip Error: %s:%d, ", __FILE__, __LINE__);               \
      fprintf(stderr, "code: %d\n", res);                                      \
      fprintf(stderr, "error: %s\n", hipGetErrorString(res));                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

hipModule_t hipModule = NULL;
// char* moduleImg = NULL;

void initHipCtx(char *moduleImg) {
  HIP_CHECK(hipInit(0));

  if (moduleImg) {
    HIP_CHECK(hipModuleLoadData(&hipModule, moduleImg));
  }
}

template <typename T> int64_t HipMalloc(int64_t size) {
  initHipCtx(NULL);
  hipDeviceptr_t device_ptr;
  HIP_CHECK(hipMalloc(&device_ptr, size * sizeof(T)));

  // return (int64_t)malloc(size * sizeof(T));
  return (int64_t)device_ptr;
}

extern "C" __attribute__((visibility("default"))) void
HipSetModuleImage(char *ptx) {
  // if (moduleImg == NULL)
  {
    // printf("Setting PTX");
    initHipCtx(ptx);
  }
}

extern "C" __attribute__((visibility("default"))) int64_t
HipMallocF64(int64_t size) {
  // printf("Allocating memory of size: %ld\n", size);
  return HipMalloc<double>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t
HipMallocF32(int64_t size) {
  // printf("Allocating memory of size: %ld\n", size);
  return HipMalloc<float>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t
HipMallocI64(int64_t size) {
  // printf("Allocating memory of size: %ld\n", size);
  return HipMalloc<int64_t>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t
HipMallocI32(int64_t size) {
  // printf("Allocating memory of size: %ld\n", size);
  return HipMalloc<int32_t>(size);
}

extern "C" __attribute__((visibility("default"))) void HipFree(int64_t ptr) {
  initHipCtx(NULL);

  // printf("Freeing memory\n");
  HIP_CHECK(hipFree((void *)ptr));
  // free((void*)ptr);
}

template <typename T>
void HipMemcpy(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
               int64_t size, int64_t stride, int64_t direction) {
  initHipCtx(NULL);

  // printf("Memcpy memory of size: %ld\n", size);

  if (direction == 0) // Host to Device
  {
    HIP_CHECK(hipMemcpyHtoD((void *)device, aligned_ptr, size * sizeof(T)));
    // memcpy((void*)device, aligned_ptr, size * sizeof(T));
  } else // Device to Host
  {
    HIP_CHECK(hipMemcpyDtoH(aligned_ptr, (void *)device, size * sizeof(T)));
    // memcpy(aligned_ptr, (void*)device, size * sizeof(T));
  }
}

extern "C" __attribute__((visibility("default"))) void
HipMemcpyIndex(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
               int64_t size, int64_t stride, int64_t direction) {
  HipMemcpy<int64_t>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void
HipMemcpyI64(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
             int64_t size, int64_t stride, int64_t direction) {
  HipMemcpy<int64_t>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void
HipMemcpyF64(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
             int64_t size, int64_t stride, int64_t direction) {
  HipMemcpy<double>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void
HipMemcpyI32(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
             int64_t size, int64_t stride, int64_t direction) {
  HipMemcpy<int32_t>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void
HipMemcpyF32(int64_t device, void *ptr, void *aligned_ptr, int64_t offset,
             int64_t size, int64_t stride, int64_t direction) {
  HipMemcpy<float>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

struct UnrankedMemRef {

  void *ptr;
  void *aligned_ptr;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

const int64_t MAX_NUM_BLOCKS_X = 2147483647;
const int64_t MAX_NUM_BLOCKS_Y = 65535;
const int64_t MAX_NUM_BLOCKS_Z = 65535;

extern "C" __attribute__((visibility("default"))) void
HipLaunchKernel(int64_t realblocksX, int64_t realblocksY, int64_t realblocksZ,
                int64_t tritonBlockX, int64_t tritonBlockY,
                int64_t tritonBlockZ, void *ptr, void *aligned_ptr,
                int64_t offset, int64_t size, int64_t stride, char *kernel,
                int64_t kernel_name_size, int64_t sharedMem, int64_t numWraps,
                int64_t threadsPerWarp) {
  hipFunction_t hipFunction = NULL;
  unsigned blocksPerGridX = std::min(realblocksX, MAX_NUM_BLOCKS_X);
  unsigned blocksPerGridY = std::min(realblocksY, MAX_NUM_BLOCKS_Y);
  unsigned blocksPerGridZ = std::min(realblocksZ, MAX_NUM_BLOCKS_Z);
  HIP_CHECK(hipModuleGetFunction(&hipFunction, hipModule, kernel));
  void **cast_args = (void **)aligned_ptr;
  HIP_CHECK(hipModuleLaunchKernel(hipFunction, blocksPerGridX, blocksPerGridY,
                                  blocksPerGridZ, numWraps * threadsPerWarp, 1,
                                  1, sharedMem, NULL, cast_args, NULL));
}