//
// Created by Zhen Peng on 10/3/25.
//

#include "comet/ExecutionEngine/DoubleBufferUtils.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <stdatomic.h>

//extern "C" void _milr_ciface_comet_sanity_check(UnrankedMemRefType<double> *A_buffer1,
//                                                UnrankedMemRefType<double> *A_buffer2,
//                                                double *A_buffer1_pointer,
//                                                double *A_buffer2_pointer)
//{
//  std::cout << __FILE__ << ":" << __LINE__ << " comet_sanity_check();" << std::endl;
//  auto A_buffer1_cast = DynamicMemRefType<double>(*A_buffer1);
//  auto A_buffer2_cast = DynamicMemRefType<double>(*A_buffer2);
//
//  std::cout << "A_buffer1:" << std::endl;
//  _print_dynamic_memref(A_buffer1_cast);
//  if (A_buffer1_cast.data == A_buffer1_pointer) {
//    std::cout << "A_buffer1 == A_buffer1_pointer" << std::endl;
//  } else {
//    std::cout << "A_buffer1 != A_buffer1_pointer" << std::endl;
//  }
//
//  std::cout << "A_buffer2:" << std::endl;
//  _print_dynamic_memref(A_buffer2_cast);
//  if (A_buffer2_cast.data == A_buffer2_pointer) {
//    std::cout << "A_buffer2 == A_buffer2_pointer" << std::endl;
//  } else {
//    std::cout << "A_buffer2 != A_buffer2_pointer" << std::endl;
//  }
//}
//
//extern "C" void comet_sanity_check(int64_t A_buffer1_rank, void *A_buffer1_data,
//                                   int64_t A_buffer2_rank, void *A_buffer2_data,
//                                   int64_t A_buffer1_pointer,
//                                   int64_t A_buffer2_pointer)
//{
//  UnrankedMemRefType<double> A_buffer1 = {A_buffer1_rank, A_buffer1_data};
//  UnrankedMemRefType<double> A_buffer2 = {A_buffer2_rank, A_buffer2_data};
//  double *A_1_pointer = (double *) A_buffer1_pointer;
//  double *A_2_pointer = (double *) A_buffer2_pointer;
//  _milr_ciface_comet_sanity_check(&A_buffer1,
//                                  &A_buffer2,
//                                  A_1_pointer,
//                                  A_2_pointer);
//}
//void _milr_ciface_comet_sanity_check(UnrankedMemRefType<UnrankedMemRefType<double>> *A_unranked,
//                                                UnrankedMemRefType<int8_t> *flag_unranked)
void _milr_ciface_comet_print_memref_to_memref_f64(UnrankedMemRefType<UnrankedMemRefType<double>> *A_unranked)
{
  std::cout << __FILE__ << ":" << __LINE__ << " comet_sanity_check();" << std::endl;
  auto A_dynamic = DynamicMemRefType<UnrankedMemRefType<double>>(*A_unranked);
  std::cout << "A_dynamic.rank: " << A_dynamic.rank << std::endl;
  for (int64_t r = 0; r < A_dynamic.rank; ++r) {
    std::cout << "A_dynamic.sizes[" << r << "]: " << A_dynamic.sizes[r] << std::endl;
  }
  for (int64_t r = 0; r < A_dynamic.rank; ++r) {
    std::cout << "A_dynamic.strides[" << r << "]: " << A_dynamic.strides[r] << std::endl;
  }

  DynamicMemRefType<double> A_inside_dynamic(A_dynamic.data[0]);
//  std::cout << "A_inside_dynamic.rank: " << A_inside_dynamic.rank << std::endl;
//  std::cout << "A_inside_dynamic.sizes[0]: " << A_inside_dynamic.sizes[0] << std::endl;
  _print_dynamic_memref(A_inside_dynamic);

}

void comet_print_memref_to_memref_f64(int64_t A_memref_size, void *A_memref)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_unranked = {A_memref_size, A_memref};
  _milr_ciface_comet_print_memref_to_memref_f64(&A_unranked);
}

int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref)
{
//  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
//  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
//  return __atomic_load_n(&flag_dynamic.data[0], __ATOMIC_ACQUIRE);

  int8_t *ptr = get_pointer_from_memref<int8_t>(memref_size, memref);
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value)
{
//  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
//  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
//  __atomic_store_n(&flag_dynamic.data[0], value, __ATOMIC_RELEASE);

  int8_t *ptr = get_pointer_from_memref<int8_t>(memref_size, memref);
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

void comet_swap_buffers(int64_t memref1_size, void *memref1, int64_t memref2_size, void *memref2)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_unranked = {memref1_size, memref1};
  DynamicMemRefType<UnrankedMemRefType<double>> A_dynamic(A_unranked);

  UnrankedMemRefType<UnrankedMemRefType<double>> B_unranked = {memref2_size, memref2};
  DynamicMemRefType<UnrankedMemRefType<double>> B_dynamic(B_unranked);

  std::swap(A_dynamic.data[0], B_dynamic.data[0]);

}



void background_process(DynamicMemRefType<UnrankedMemRefType<double>> &A_buffer1_dynamic,
                        int64_t flag_size, void *flag_memref)
{
  DynamicMemRefType<double> A_buffer1_inside_dynamic(A_buffer1_dynamic.data[0]);
  for (int64_t i = 0; i < A_buffer1_inside_dynamic.sizes[0]; ++i) {
    A_buffer1_inside_dynamic.data[i] = i + 1;
  }

  int8_t *flag = get_pointer_from_memref<int8_t>(flag_size, flag_memref);

  std::cout << "(C++) before flag: " << (int32_t) _atomic_load_n_i8(flag) << std::endl;
  while (!_atomic_load_n_i8(flag)) {
    std::cout << "(C++) spin... flag: " << (int32_t) _atomic_load_n_i8(flag) << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  _atomic_store_n_i8(flag, (int8_t) 0);
  std::cout << "(C++) after flag: " << (int32_t) _atomic_load_n_i8(flag) << std::endl;
}

void comet_initialize_test_thread(int64_t A_memref1_size, void *A_memref1,
                                  int64_t flag_size, void *flag_memref)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1_unranked = {A_memref1_size, A_memref1};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1_dynamic(A_buffer1_unranked);

  std::thread background_thread(background_process,
                                std::ref(A_buffer1_dynamic),
                                flag_size, flag_memref);
//  background_thread.join();
  background_thread.detach();
}

void comet_async_background_thread_test(int64_t A_memref1_size, void *A_memref1,
                                        int64_t flag_size, void *flag_memref)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1_unranked = {A_memref1_size, A_memref1};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1_dynamic(A_buffer1_unranked);

  background_process(A_buffer1_dynamic,
                     flag_size, flag_memref);
}


int8_t _atomic_load_n_i8(int8_t *ptr)
{
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void _atomic_store_n_i8(int8_t *ptr, int8_t val)
{
  __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
}

void _copy_to_buffer(double *A,
                     int64_t A1,
                     int64_t A2,
                     int64_t A1_offset,
                     int64_t A2_offset,
                     double *B,
                     int64_t B1,
                     int64_t B2,
                     int64_t block_rows,
                     int64_t block_cols)
{
  for (int64_t i = 0; i < block_rows; ++i) {
    int64_t a_i = A1_offset + i;
    for (int64_t j = 0; j < block_cols; ++j) {
      int64_t a_j = A2_offset + j;
      B[i * B2 + j] = A[a_i * A2 + a_j];
    }
  }
}

//void _swap_buffer(double **buffer1, double **buffer2)
//{
//  std::swap(*buffer1, *buffer2);
//}

/// Do swap on the auxiliary side
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
    /* int8_t *B_buffer_is_ready */ int64_t B_buffer_flag_memref_size, void *B_buffer_flag_memref)
{
  double *A = get_pointer_from_memref<double>(A_memref_size, A_memref);
  double *B = get_pointer_from_memref<double>(B_memref_size, B_memref);
  int8_t *A_buffer_is_ready = get_pointer_from_memref<int8_t>(A_buffer_flag_memref_size, A_buffer_flag_memref);
  int8_t *B_buffer_is_ready = get_pointer_from_memref<int8_t>(B_buffer_flag_memref_size, B_buffer_flag_memref);
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1_unranked = {A_buffer1_mm_size, A_buffer1_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1_dynamic(A_buffer1_unranked);
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer2_unranked = {A_buffer2_mm_size, A_buffer2_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer2_dynamic(A_buffer2_unranked);
  UnrankedMemRefType<UnrankedMemRefType<double>> B_buffer1_unranked = {B_buffer1_mm_size, B_buffer1_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> B_buffer1_dynamic(B_buffer1_unranked);
  UnrankedMemRefType<UnrankedMemRefType<double>> B_buffer2_unranked = {B_buffer2_mm_size, B_buffer2_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> B_buffer2_dynamic(B_buffer2_unranked);

//  /// test
//  std::cout << "test on C++." << std::endl;
//  std::this_thread::sleep_for(std::chrono::seconds(10));
//  _atomic_store_n_i8(A_buffer_is_ready, (int8_t) 1);
//  return;
//  /// end test

//  for (int64_t ii = 0; ii < A1; ii += A1_tile) {
//    int64_t A_block_rows = std::min(A1_tile, A1 - ii);
//    for (int64_t kk = 0; kk < A2; kk += A2_tile) {
//      int64_t A_block_cols = std::min(A2_tile, A2 - kk);
//      int64_t B_block_rows = A_block_cols;
//
//      DynamicMemRefType<double> A_buffer2_inside_dynamic(A_buffer2_dynamic.data[0]);
//      double *A_buffer2 = A_buffer2_inside_dynamic.data;
////      _copy_to_buffer(A,
////                      A1, A2,
////                      /*A1_offset=*/ii, /*A2_offset=*/kk,
////                      A_buffer2,
////                      A1_tile, A2_tile,
////                      A_block_rows, A_block_cols);
//////      while (A_buffer_is_ready->load(std::memory_order_acquire)) {
////      while (_atomic_load_n_i8(A_buffer_is_ready)) {
////        /// test
////        std::cout << "A_buffer_is_ready. Spin in C++." << std::endl;
////        std::this_thread::sleep_for(std::chrono::seconds(1));
////        /// end test
////        ;  /// Spin
////      }
//////      _swap_buffer(A_buffer1, A_buffer2);
////      _swap_buffer(A_buffer1_dynamic, A_buffer2_dynamic);
//////      A_buffer_is_ready->store(true, std::memory_order_release);
////      _atomic_store_n_i8(A_buffer_is_ready, (int8_t) 1);
//
////      for (int64_t jj = 0; jj < B2; jj += B2_tile) {
////        int64_t B_block_cols = std::min(B2_tile, B2 - jj);
////
////        DynamicMemRefType<double> B_buffer2_inside_dynamic(B_buffer2_dynamic.data[0]);
////        double *B_buffer2 = B_buffer2_inside_dynamic.data;
////        _copy_to_buffer(B,
////                        B1, B2,
////                        /*B1_offset=*/kk, /*B2_offset=*/jj,
////                        B_buffer2,
////                        B1_tile, B2_tile,
////                        B_block_rows, B_block_cols);
//////        while (B_buffer_is_ready->load(std::memory_order_acquire)) {
////        while (_atomic_load_n_i8(B_buffer_is_ready)) {
////          /// test
////          std::cout << "B_buffer_is_ready. Spin in C++." << std::endl;
////          std::this_thread::sleep_for(std::chrono::seconds(1));
////          /// end test
////          ; /// Spin
////        }
//////        _swap_buffer(B_buffer1, B_buffer2);
////        _swap_buffer(B_buffer1_dynamic, B_buffer2_dynamic);
//////        B_buffer_is_ready->store(true, std::memory_order_release);
////        _atomic_store_n_i8(B_buffer_is_ready, (int8_t) 1);
////      }
//    }
//  }
}

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
    /* int8_t *B_buffer_is_ready */ int64_t B_buffer_flag_memref_size, void *B_buffer_flag_memref)
{
//  std::thread background_thread(background_process, std::ref(A_buffer1_dynamic));
//  background_thread.join();
  std::thread background_thread(
  /* function_name */             _background_process_v1,
  /* double *A */                 A_memref_size, A_memref,
  /* uint64_t A1 */               A1,
  /* uint64_t A2 */               A2,
  /* double **A_buffer1 */        A_buffer1_mm_size, A_buffer1_mm,
  /* double **A_buffer2 */        A_buffer2_mm_size, A_buffer2_mm,
  /* uint64_t A1_tile */          A1_tile,
  /* uint64_t A2_tile */          A2_tile,
  /* int8_t *A_buffer_is_ready */ A_buffer_flag_memref_size, A_buffer_flag_memref,
  /* double *B */                 B_memref_size, B_memref,
  /* uint64_t B1 */               B1,
  /* uint64_t B2 */               B2,
  /* double **B_buffer1 */        B_buffer1_mm_size, B_buffer1_mm,
  /* double **B_buffer2 */        B_buffer2_mm_size, B_buffer2_mm,
  /* uint64_t B1_tile */          B1_tile,
  /* uint64_t B2_tile */          B2_tile,
  /* int8_t *B_buffer_is_ready */ B_buffer_flag_memref_size, B_buffer_flag_memref);
  background_thread.join();
}