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

void comet_initialize_test_thread(int64_t A_memref1_size, void *A_memref1,
                                  int64_t flag_size, void *flag_memref);
void comet_async_background_thread_test(int64_t A_memref1_size, void *A_memref1,
                                        int64_t flag_size, void *flag_memref);
void comet_initialize_double_buffer_thread(
    /* double *A */                 int64_t A_memref_size, void *A_memref,
    /* uint64_t A1 */               int64_t A1,
    /* uint64_t A2 */               int64_t A2,
    /* double **A_buffer1 */        int64_t A_buffer1_mm_size, void *A_buffer1_mm,
    /* double **A_buffer2 */        int64_t A_buffer2_mm_size, void *A_buffer2_mm,
    /* uint64_t A1_tile */          int64_t A1_tile,
    /* uint64_t A2_tile */          int64_t A2_tile,
    /* int8_t *A_buffer_is_ready */ int64_t A_buffer_flag_memref_size, void *A_buffer_flag_memref,
    /* double *B */                 int64_t B_memref_size, void *B_memref,
    /* uint64_t B1 */               int64_t B1,
    /* uint64_t B2 */               int64_t B2,
    /* double **B_buffer1 */        int64_t B_buffer1_mm_size, void *B_buffer1_mm,
    /* double **B_buffer2 */        int64_t B_buffer2_mm_size, void *B_buffer2_mm,
    /* uint64_t B1_tile */          int64_t B1_tile,
    /* uint64_t B2_tile */          int64_t B2_tile,
    /* int8_t *B_buffer_is_ready */ int64_t B_buffer_flag_memref_size, void *B_buffer_flag_memref);
void comet_print_memref_to_memref_f64(int64_t A_memref_size, void *A_memref);
int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref);
void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value);
void comet_swap_buffers(int64_t memref1_size, void *memref1, int64_t memref2_size, void *memref2);

#ifdef __cplusplus
};
#endif

int8_t _atomic_load_n_i8(int8_t *ptr);
void _atomic_store_n_i8(int8_t *ptr, int8_t val);
void _copy_to_buffer(double *A,
                     int64_t A1,
                     int64_t A2,
                     int64_t A1_offset,
                     int64_t A2_offset,
                     double *B,
                     int64_t B1,
                     int64_t B2,
                     int64_t block_rows,
                     int64_t block_cols);
void _background_process_v1(
    /* double *A */                 int64_t A_memref_size, void *A_memref,
    /* uint64_t A1 */               int64_t A1,
    /* uint64_t A2 */               int64_t A2,
    /* double **A_buffer1 */        int64_t A_buffer1_mm_size, void *A_buffer1_mm,
    /* double **A_buffer2 */        int64_t A_buffer2_mm_size, void *A_buffer2_mm,
    /* uint64_t A1_tile */          int64_t A1_tile,
    /* uint64_t A2_tile */          int64_t A2_tile,
    /* int8_t *A_buffer_is_ready */ int64_t A_buffer_flag_memref_size, void *A_buffer_flag_memref,
    /* double *B */                 int64_t B_memref_size, void *B_memref,
    /* uint64_t B1 */               int64_t B1,
    /* uint64_t B2 */               int64_t B2,
    /* double **B_buffer1 */        int64_t B_buffer1_mm_size, void *B_buffer1_mm,
    /* double **B_buffer2 */        int64_t B_buffer2_mm_size, void *B_buffer2_mm,
    /* uint64_t B1_tile */          int64_t B1_tile,
    /* uint64_t B2_tile */          int64_t B2_tile,
    /* int8_t *B_buffer_is_ready */ int64_t B_buffer_flag_memref_size, void *B_buffer_flag_memref);

template <typename T>
T *get_pointer_from_memref(int64_t memref_size, void *memref)
{
  UnrankedMemRefType<T> memref_unranked = {memref_size, memref};
  DynamicMemRefType<T> memref_dynamic(memref_unranked);

  return memref_dynamic.data;
}

template <typename T>
void _swap_buffer(DynamicMemRefType<UnrankedMemRefType<T>> &buffer1,
                  DynamicMemRefType<UnrankedMemRefType<T>> &buffer2)
{
  std::swap(buffer1.data[0], buffer2.data[0]);
}


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
