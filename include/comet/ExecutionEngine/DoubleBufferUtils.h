//
// Created by Zhen Peng on 10/3/25.
//

#ifndef COMET_DOUBLEBUFFERUTILS_H
#define COMET_DOUBLEBUFFERUTILS_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

void comet_initialize_double_buffer_thread(int64_t A_memref1_size, void *A_memref1);
//void comet_sanity_check(int64_t A_memref_size, void *A_memref, int64_t flag_size, void *flag_memref);
void comet_print_memref_to_memref_f64(int64_t A_memref_size, void *A_memref);
int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref);
void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value);
void comet_swap_buffers(int64_t memref1_size, void *memref1, int64_t memref2_size, void *memref2);

#ifdef __cplusplus
};
#endif

template <typename T>
void _print_dynamic_memref(const DynamicMemRefType<T> &M)
{
  std::cout << "M.rank: " << M.rank << std::endl;
  for (int64_t r = 0; r < M.rank; ++r) {
    std::cout << "M.sizes[" << r << "]: " << M.sizes[r] << std::endl;
  }
  for (int64_t r = 0; r < M.rank; ++r) {
    std::cout << "M.strides[" << r << "]: " << M.strides[r] << std::endl;
  }

  int64_t size = 1;
  for (int64_t r = 0; r < M.rank; ++r) {
    size *= M.sizes[r];
  }

  for (int64_t i = 0; i < size; ++i) {
    std::cout << M.data[i] << " ";
  }
  std::cout << std::endl;
}


#endif //COMET_DOUBLEBUFFERUTILS_H
