//
// Created by Zhen Peng on 10/3/25.
//

#include "comet/ExecutionEngine/DoubleBufferUtils.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
//#include <stdatomic.h>
/// Ref: https://github.com/python/cpython/issues/67832#issuecomment-1093677548
#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#include <mutex>
std::mutex log_mutex;

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


void comet_print_matrix_f64(int64_t A_memref_size, void *A_memref, int64_t A1, int64_t A2)
{
  double *data = _get_pointer_from_memref<double>(A_memref_size, A_memref);
//  std::cout << "memref_size: " << A_memref_size << std::endl;
  for (int64_t i = 0; i < A1; ++i) {
    for (int64_t j = 0; j < A2; ++j) {
      std::cout << data[i * A2 + j] << " ";
    }
    std::cout << std::endl;
  }
}

int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref)
{
//  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
//  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
//  return __atomic_load_n(&flag_dynamic.data[0], __ATOMIC_ACQUIRE);

  int8_t *ptr = _get_pointer_from_memref<int8_t>(memref_size, memref);
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value)
{
//  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
//  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
//  __atomic_store_n(&flag_dynamic.data[0], value, __ATOMIC_RELEASE);

  int8_t *ptr = _get_pointer_from_memref<int8_t>(memref_size, memref);
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

  int8_t *flag = _get_pointer_from_memref<int8_t>(flag_size, flag_memref);

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
//  /// test
//  {
//    if (A1_offset == 4) {
//      std::lock_guard<std::mutex> lock(log_mutex);
//      std::cout << "111 After:" << std::endl;
//      std::cout << "!!! B(A_buffer2):" << std::endl;
//      _print_matrix_buffer(B, B1, B2, block_rows, block_cols);
//      std::cout << "!!! A(A):"  << std::endl;
//      _print_matrix_buffer(A, A1, A2, A1, A2);
//      std::cout << "-----------------" << std::endl;
//    }
//  }
//  /// end test

  for (int64_t i = 0; i < block_rows; ++i) {
    int64_t a_i = A1_offset + i;
    for (int64_t j = 0; j < block_cols; ++j) {
      int64_t a_j = A2_offset + j;
      B[i * B2 + j] = A[a_i * A2 + a_j];
    }
  }

//  /// test
//  {
//    if (A1_offset == 4) {
//      std::lock_guard<std::mutex> lock(log_mutex);
//      std::cout << "222 After:" << std::endl;
//      std::cout << "!!! B(A_buffer2):" << std::endl;
//      _print_matrix_buffer(B, B1, B2, block_rows, block_cols);
//      std::cout << "!!! A(A):"  << std::endl;
//      _print_matrix_buffer(A, A1, A2, A1, A2);
//      std::cout << "-----------------" << std::endl;
//    }
//  }
//  /// end test
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
  double *A = _get_pointer_from_memref<double>(A_memref_size, A_memref);
  double *B = _get_pointer_from_memref<double>(B_memref_size, B_memref);
  int8_t *A_buffer_is_ready = _get_pointer_from_memref<int8_t>(A_buffer_flag_memref_size, A_buffer_flag_memref);
  int8_t *B_buffer_is_ready = _get_pointer_from_memref<int8_t>(B_buffer_flag_memref_size, B_buffer_flag_memref);
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

  for (int64_t ii = 0; ii < A1; ii += A1_tile) {
    int64_t A_block_rows = std::min(A1_tile, A1 - ii);
    for (int64_t kk = 0; kk < A2; kk += A2_tile) {
      int64_t A_block_cols = std::min(A2_tile, A2 - kk);
      int64_t B_block_rows = A_block_cols;

      DynamicMemRefType<double> A_buffer2_inside_dynamic(A_buffer2_dynamic.data[0]);
      double *A_buffer2 = A_buffer2_inside_dynamic.data;
      _copy_to_buffer(A,
                      A1, A2,
                      /*A1_offset=*/ii, /*A2_offset=*/kk,
                      A_buffer2,
                      A1_tile, A2_tile,
                      A_block_rows, A_block_cols);
//      while (A_buffer_is_ready->load(std::memory_order_acquire)) {
      while (_atomic_load_n_i8(A_buffer_is_ready)) {
//        /// test
//        std::cout << "A_buffer_is_ready. Spin in C++." << std::endl;
//        std::this_thread::sleep_for(std::chrono::milliseconds(100));
//        /// end test
        ;  /// Spin
      }
//      _swap_buffer(A_buffer1, A_buffer2);
      _swap_buffer(A_buffer1_dynamic, A_buffer2_dynamic);
//      A_buffer_is_ready->store(true, std::memory_order_release);
      _atomic_store_n_i8(A_buffer_is_ready, (int8_t) 1);

      for (int64_t jj = 0; jj < B2; jj += B2_tile) {
        int64_t B_block_cols = std::min(B2_tile, B2 - jj);

        DynamicMemRefType<double> B_buffer2_inside_dynamic(B_buffer2_dynamic.data[0]);
        double *B_buffer2 = B_buffer2_inside_dynamic.data;
        _copy_to_buffer(B,
                        B1, B2,
                        /*B1_offset=*/kk, /*B2_offset=*/jj,
                        B_buffer2,
                        B1_tile, B2_tile,
                        B_block_rows, B_block_cols);
//        while (B_buffer_is_ready->load(std::memory_order_acquire)) {
        while (_atomic_load_n_i8(B_buffer_is_ready)) {
//          /// test
//          std::cout << "B_buffer_is_ready. Spin in C++." << std::endl;
//          std::this_thread::sleep_for(std::chrono::milliseconds(100));
//          /// end test
          ; /// Spin
        }
//        _swap_buffer(B_buffer1, B_buffer2);
        _swap_buffer(B_buffer1_dynamic, B_buffer2_dynamic);
//        B_buffer_is_ready->store(true, std::memory_order_release);
        _atomic_store_n_i8(B_buffer_is_ready, (int8_t) 1);
      }
    }
  }
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
//  /// Use C++ thread and detach();
//  std::thread background_thread(
//  /* function_name */             _background_process_v1,
//  /* double *A */                 A_memref_size, A_memref,
//  /* uint64_t A1 */               A1,
//  /* uint64_t A2 */               A2,
//  /* double **A_buffer1 */        A_buffer1_mm_size, A_buffer1_mm,
//  /* double **A_buffer2 */        A_buffer2_mm_size, A_buffer2_mm,
//  /* uint64_t A1_tile */          A1_tile,
//  /* uint64_t A2_tile */          A2_tile,
//  /* int8_t *A_buffer_is_ready */ A_buffer_flag_memref_size, A_buffer_flag_memref,
//  /* double *B */                 B_memref_size, B_memref,
//  /* uint64_t B1 */               B1,
//  /* uint64_t B2 */               B2,
//  /* double **B_buffer1 */        B_buffer1_mm_size, B_buffer1_mm,
//  /* double **B_buffer2 */        B_buffer2_mm_size, B_buffer2_mm,
//  /* uint64_t B1_tile */          B1_tile,
//  /* uint64_t B2_tile */          B2_tile,
//  /* int8_t *B_buffer_is_ready */ B_buffer_flag_memref_size, B_buffer_flag_memref);
//  background_thread.detach();

  /// Called in MLIR by async dialect, so no C++ thread is needed.
  _background_process_v1(
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
}

void comet_test_array_of_buffers(
    uint64_t num_compute_workers,
    uint64_t A1_tile, uint64_t A2_tile,
    int64_t A_buffer1s_size, void *A_buffer1s)
{
  UnrankedMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer1s_unranked = {A_buffer1s_size, A_buffer1s};
  DynamicMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer1s_dynamic(A_buffer1s_unranked);

  for (uint64_t i = 0; i < num_compute_workers; ++i) {
    UnrankedMemRefType<UnrankedMemRefType<double>> buffer_ptr_unranked = A_buffer1s_dynamic.data[i];
    DynamicMemRefType<UnrankedMemRefType<double>> buffer_ptr_dynamic(buffer_ptr_unranked);
    UnrankedMemRefType<double> buffer_unranked = buffer_ptr_dynamic.data[0];
    DynamicMemRefType<double> buffer_dynamic(buffer_unranked);
    double *buffer = buffer_dynamic.data;
    std::cout << "worker: " << i << " ";
    _print_matrix_buffer(buffer, A1_tile, A2_tile, A1_tile, A2_tile);
  }
}

void comet_test_write_to_buffer(
    uint64_t val,
    uint64_t A1_tile, uint64_t A2_tile,
    int64_t A_buffer_size, void *A_buffer)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> buffer_ptr_unranked = {A_buffer_size, A_buffer};
  DynamicMemRefType<UnrankedMemRefType<double>> buffer_ptr_dynamic(buffer_ptr_unranked);
  UnrankedMemRefType<double> buffer_unranked = buffer_ptr_dynamic.data[0];
  DynamicMemRefType<double> buffer_dynamic(buffer_unranked);
  double *buffer = buffer_dynamic.data;
  for (uint64_t i = 0; i < A1_tile; ++i) {
    for (uint64_t j = 0; j < A2_tile; ++j) {
      buffer[i * A2_tile + j] = (double) val;
    }
  }
}

/// -------------------------------------------------------------------------------
/// Parallel Double Buffering: multiple compute workers and less auxiliary workers
/// -------------------------------------------------------------------------------
/// Compute worker
void comet_double_buffer_compute_worker_drive(
    /* uint64_t A1 */                uint64_t A1,
    /* uint64_t A2 */                uint64_t A2,
    /* uint64_t A1_tile */           uint64_t A1_tile,
    /* uint64_t A2_tile */           uint64_t A2_tile,
    /* uint64_t *A1_offset_ptr */    int64_t A1_offset_ptr_m_size, void *A1_offset_ptr_m,
    /* uint64_t *A2_offset_ptr */    int64_t A2_offset_ptr_m_size, void *A2_offset_ptr_m,
    /* uint64_t *A_block_rows_ptr */ int64_t A_block_rows_ptr_m_size, void *A_block_rows_ptr_m,
    /* uint64_t *A_block_cols_ptr */ int64_t A_block_cols_ptr_m_size, void *A_block_cols_ptr_m,
    /* double **A_buffer1 */         int64_t A_buffer1_mm_size, void *A_buffer1_mm,
    /* int8_t *A_buffer_is_ready */  int64_t A_buffer_is_ready_m_size, void *A_buffer_is_ready_m,
    /* uint64_t B1 */                uint64_t B1,
    /* uint64_t B2 */                uint64_t B2,
    /* uint64_t B1_tile */           uint64_t B1_tile,
    /* uint64_t B2_tile */           uint64_t B2_tile,
    /* uint64_t *B1_offset_ptr */    int64_t B1_offset_ptr_m_size, void *B1_offset_ptr_m,
    /* uint64_t *B2_offset_ptr */    int64_t B2_offset_ptr_m_size, void *B2_offset_ptr_m,
    /* uint64_t *B_block_rows_ptr */ int64_t B_block_rows_ptr_m_size, void *B_block_rows_ptr_m,
    /* uint64_t *B_block_cols_ptr */ int64_t B_block_cols_ptr_m_size, void *B_block_cols_ptr_m,
    /* double **B_buffer1 */         int64_t B_buffer1_mm_size, void *B_buffer1_mm,
    /* int8_t *B_buffer_is_ready */  int64_t B_buffer_is_ready_m_size, void *B_buffer_is_ready_m,
    /* double *C */                  int64_t C_m_size, void *C_m,
    /* uint64_t i_start */           uint64_t i_start,
    /* uint64_t num_local_i_tiles */ uint64_t num_local_i_tiles,
    /* int8_t *is_finished */        int64_t is_finished_m_size, void *is_finished_m)
{
//  std::thread bg_thread(
//      _bg_compute_worker_drive,
//      A1,
//      A2,
//      A1_tile,
//      A2_tile,
//      A1_offset_ptr_m_size, A1_offset_ptr_m,
//      A2_offset_ptr_m_size, A2_offset_ptr_m,
//      A_block_rows_ptr_m_size, A_block_rows_ptr_m,
//      A_block_cols_ptr_m_size, A_block_cols_ptr_m,
//      A_buffer1_mm_size, A_buffer1_mm,
//      A_buffer_is_ready_m_size, A_buffer_is_ready_m,
//      B1,
//      B2,
//      B1_tile,
//      B2_tile,
//      B1_offset_ptr_m_size, B1_offset_ptr_m,
//      B2_offset_ptr_m_size, B2_offset_ptr_m,
//      B_block_rows_ptr_m_size, B_block_rows_ptr_m,
//      B_block_cols_ptr_m_size, B_block_cols_ptr_m,
//      B_buffer1_mm_size, B_buffer1_mm,
//      B_buffer_is_ready_m_size, B_buffer_is_ready_m,
//      C_m_size, C_m,
//      i_start,
//      num_local_i_tiles,
//      is_finished_m_size, is_finished_m);
//  bg_thread.detach();
  _bg_compute_worker_drive(
      A1,
      A2,
      A1_tile,
      A2_tile,
      A1_offset_ptr_m_size, A1_offset_ptr_m,
      A2_offset_ptr_m_size, A2_offset_ptr_m,
      A_block_rows_ptr_m_size, A_block_rows_ptr_m,
      A_block_cols_ptr_m_size, A_block_cols_ptr_m,
      A_buffer1_mm_size, A_buffer1_mm,
      A_buffer_is_ready_m_size, A_buffer_is_ready_m,
      B1,
      B2,
      B1_tile,
      B2_tile,
      B1_offset_ptr_m_size, B1_offset_ptr_m,
      B2_offset_ptr_m_size, B2_offset_ptr_m,
      B_block_rows_ptr_m_size, B_block_rows_ptr_m,
      B_block_cols_ptr_m_size, B_block_cols_ptr_m,
      B_buffer1_mm_size, B_buffer1_mm,
      B_buffer_is_ready_m_size, B_buffer_is_ready_m,
      C_m_size, C_m,
      i_start,
      num_local_i_tiles,
      is_finished_m_size, is_finished_m);
}

void _bg_compute_worker_drive(
    /* uint64_t A1 */                uint64_t A1,
    /* uint64_t A2 */                uint64_t A2,
    /* uint64_t A1_tile */           uint64_t A1_tile,
    /* uint64_t A2_tile */           uint64_t A2_tile,
    /* uint64_t *A1_offset_ptr */    int64_t A1_offset_ptr_m_size, void *A1_offset_ptr_m,
    /* uint64_t *A2_offset_ptr */    int64_t A2_offset_ptr_m_size, void *A2_offset_ptr_m,
    /* uint64_t *A_block_rows_ptr */ int64_t A_block_rows_ptr_m_size, void *A_block_rows_ptr_m,
    /* uint64_t *A_block_cols_ptr */ int64_t A_block_cols_ptr_m_size, void *A_block_cols_ptr_m,
    /* double **A_buffer1 */         int64_t A_buffer1_mm_size, void *A_buffer1_mm,
    /* int8_t *A_buffer_is_ready */  int64_t A_buffer_is_ready_m_size, void *A_buffer_is_ready_m,
    /* uint64_t B1 */                uint64_t B1,
    /* uint64_t B2 */                uint64_t B2,
    /* uint64_t B1_tile */           uint64_t B1_tile,
    /* uint64_t B2_tile */           uint64_t B2_tile,
    /* uint64_t *B1_offset_ptr */    int64_t B1_offset_ptr_m_size, void *B1_offset_ptr_m,
    /* uint64_t *B2_offset_ptr */    int64_t B2_offset_ptr_m_size, void *B2_offset_ptr_m,
    /* uint64_t *B_block_rows_ptr */ int64_t B_block_rows_ptr_m_size, void *B_block_rows_ptr_m,
    /* uint64_t *B_block_cols_ptr */ int64_t B_block_cols_ptr_m_size, void *B_block_cols_ptr_m,
    /* double **B_buffer1 */         int64_t B_buffer1_mm_size, void *B_buffer1_mm,
    /* int8_t *B_buffer_is_ready */  int64_t B_buffer_is_ready_m_size, void *B_buffer_is_ready_m,
    /* double *C */                  int64_t C_m_size, void *C_m,
    /* uint64_t i_start */           uint64_t i_start,
    /* uint64_t num_local_i_tiles */ uint64_t num_local_i_tiles,
    /* int8_t *is_finished */        int64_t is_finished_m_size, void *is_finished_m)
{
  uint64_t *A1_offset_ptr = _get_pointer_from_memref<uint64_t>(A1_offset_ptr_m_size, A1_offset_ptr_m);
  uint64_t *A2_offset_ptr = _get_pointer_from_memref<uint64_t>(A2_offset_ptr_m_size, A2_offset_ptr_m);
  uint64_t *A_block_rows_ptr = _get_pointer_from_memref<uint64_t>(A_block_rows_ptr_m_size, A_block_rows_ptr_m);
  uint64_t *A_block_cols_ptr = _get_pointer_from_memref<uint64_t>(A_block_cols_ptr_m_size, A_block_cols_ptr_m);
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1_mm_unranked = {A_buffer1_mm_size, A_buffer1_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1_mm_dynamic(A_buffer1_mm_unranked);
  int8_t *A_buffer_is_ready = _get_pointer_from_memref<int8_t>(A_buffer_is_ready_m_size, A_buffer_is_ready_m);
  uint64_t *B1_offset_ptr = _get_pointer_from_memref<uint64_t>(B1_offset_ptr_m_size, B1_offset_ptr_m);
  uint64_t *B2_offset_ptr = _get_pointer_from_memref<uint64_t>(B2_offset_ptr_m_size, B2_offset_ptr_m);
  uint64_t *B_block_rows_ptr = _get_pointer_from_memref<uint64_t>(B_block_rows_ptr_m_size, B_block_rows_ptr_m);
  uint64_t *B_block_cols_ptr = _get_pointer_from_memref<uint64_t>(B_block_cols_ptr_m_size, B_block_cols_ptr_m);
  UnrankedMemRefType<UnrankedMemRefType<double>> B_buffer1_mm_unranked = {B_buffer1_mm_size, B_buffer1_mm};
  DynamicMemRefType<UnrankedMemRefType<double>> B_buffer1_mm_dynamic(B_buffer1_mm_unranked);
  int8_t *B_buffer_is_ready = _get_pointer_from_memref<int8_t>(B_buffer_is_ready_m_size, B_buffer_is_ready_m);
  double *C = _get_pointer_from_memref<double>(C_m_size, C_m);
  int8_t *is_finished = _get_pointer_from_memref<int8_t>(is_finished_m_size, is_finished_m);

  for (uint64_t local_i = 0; local_i < num_local_i_tiles; ++local_i) {
    uint64_t ii = i_start + local_i * A1_tile;
    uint64_t A_block_rows = std::min(A1_tile, A1 - ii);
    *A1_offset_ptr = ii;
    *A_block_rows_ptr = A_block_rows;
    for (uint64_t kk = 0; kk < A2; kk += A2_tile) {
      uint64_t A_block_cols = std::min(A2_tile, A2 - kk);
      uint64_t B_block_rows = A_block_cols;
      *A2_offset_ptr = kk;
      *B1_offset_ptr = kk;
      *A_block_cols_ptr = A_block_cols;
      *B_block_rows_ptr = B_block_rows;
      /// Sync A_buffer
//      A_buffer_is_ready->store(false, std::memory_order_release);
      _atomic_store_n_i8(A_buffer_is_ready, (int8_t) 0);
//      while (!A_buffer_is_ready->load(std::memory_order_acquire)) {
      while (!_atomic_load_n_i8(A_buffer_is_ready)) {
        ; /// Spin
      }
      for (uint64_t jj = 0; jj < B2; jj += B2_tile) {
        uint64_t B_block_cols = std::min(B2_tile, B2 - jj);
        *B2_offset_ptr = jj;
        *B_block_cols_ptr = B_block_cols;
        /// Sync B_buffer
//        B_buffer_is_ready->store(false, std::memory_order_release);
        _atomic_store_n_i8(B_buffer_is_ready, (int8_t) 0);
//        while (!B_buffer_is_ready->load(std::memory_order_acquire)) {
        while (!_atomic_load_n_i8(B_buffer_is_ready)) {
          ; /// Spin
        }
        _do_gemm_on_buffer(
            /*A_buffer_active=*/A_buffer1_mm_dynamic,
            A1_tile, A2_tile,
            A_block_rows, A_block_cols,
            /*B_buffer_active=*/B_buffer1_mm_dynamic,
            B1_tile, B2_tile,
            B_block_rows, B_block_cols,
            C,
            /*C1=*/A1, /*C2=*/B2,
            /*C1_offset=*/ii, /*C2_offset=*/jj);
//        B_buffer_is_ready->store(false, std::memory_order_release);
        _atomic_store_n_i8(B_buffer_is_ready, (int8_t) 0);
      }
//      A_buffer_is_ready->store(false, std::memory_order_release);
      _atomic_store_n_i8(A_buffer_is_ready, (int8_t) 0);
    }
  }

//  is_finished->store(true, std::memory_order_release);
  _atomic_store_n_i8(is_finished, (int8_t) 1);
}

/// Auxiliary worker
void comet_double_buffer_aux_worker_pull(
    /* double *A */                                       int64_t A_m_size, void *A_m,
    /* uint64_t A1 */                                     uint64_t A1,
    /* uint64_t A2 */                                     uint64_t A2,
    /* uint64_t A1_tile */                                uint64_t A1_tile,
    /* uint64_t A2_tile */                                uint64_t A2_tile,
    /* std::vector<double *> &A_buffer1s */               int64_t A_buffer1s_mmm_size, void *A_buffer1s_mmm,
    /* std::vector<double *> &A_buffer2s */               int64_t A_buffer2s_mmm_size, void *A_buffer2s_mmm,
    /* std::vector<uint64_t> &A1_offset_list */           int64_t A1_offset_list_mm_size, void *A1_offset_list_mm,
    /* std::vector<uint64_t> &A2_offset_list */           int64_t A2_offset_list_mm_size, void *A2_offset_list_mm,
    /* std::vector<uint64_t> &A_block_rows_list */        int64_t A_block_rows_list_mm_size, void *A_block_rows_list_mm,
    /* std::vector<uint64_t> &A_block_cols_list */        int64_t A_block_cols_list_mm_size, void *A_block_cols_list_mm,
    /* std::vector<int8_t *> &A_buffer_readys */          int64_t A_buffer_readys_mm_size, void *A_buffer_readys_mm,
    /* double *B */                                       int64_t B_m_size, void *B_m,
    /* uint64_t B1 */                                     uint64_t B1,
    /* uint64_t B2 */                                     uint64_t B2,
    /* uint64_t B1_tile */                                uint64_t B1_tile,
    /* uint64_t B2_tile */                                uint64_t B2_tile,
    /* std::vector<double *> &B_buffer1s */               int64_t B_buffer1s_mmm_size, void *B_buffer1s_mmm,
    /* std::vector<double *> &B_buffer2s */               int64_t B_buffer2s_mmm_size, void *B_buffer2s_mmm,
    /* std::vector<uint64_t> &B1_offset_list */           int64_t B1_offset_list_mm_size, void *B1_offset_list_mm,
    /* std::vector<uint64_t> &B2_offset_list */           int64_t B2_offset_list_mm_size, void *B2_offset_list_mm,
    /* std::vector<uint64_t> &B_block_rows_list */        int64_t B_block_rows_list_mm_size, void *B_block_rows_list_mm,
    /* std::vector<uint64_t> &B_block_cols_list */        int64_t B_block_cols_list_mm_size, void *B_block_cols_list_mm,
    /* std::vector<int8_t *> &B_buffer_readys */          int64_t B_buffer_readys_mm_size, void *B_buffer_readys_mm,
    /* uint64_t compute_worker_start */                   uint64_t compute_worker_start,
    /* uint64_t num_local_compute_workers */              uint64_t num_local_compute_workers,
    /* std::vector<int8_t *> &compute_workers_finished */ int64_t compute_workers_finished_mm_size, void *compute_workers_finished_mm,
    /* uint64_t num_compute_workers */                    uint64_t num_compute_workers)
{
//  std::thread bg_thread(
//      _bg_aux_worker_pull,
//      A_m_size, A_m,
//      A1,
//      A2,
//      A1_tile,
//      A2_tile,
//      A_buffer1s_mmm_size, A_buffer1s_mmm,
//      A_buffer2s_mmm_size, A_buffer2s_mmm,
//      A1_offset_list_mm_size, A1_offset_list_mm,
//      A2_offset_list_mm_size, A2_offset_list_mm,
//      A_block_rows_list_mm_size, A_block_rows_list_mm,
//      A_block_cols_list_mm_size, A_block_cols_list_mm,
//      A_buffer_readys_mm_size, A_buffer_readys_mm,
//      B_m_size, B_m,
//      B1,
//      B2,
//      B1_tile,
//      B2_tile,
//      B_buffer1s_mmm_size, B_buffer1s_mmm,
//      B_buffer2s_mmm_size, B_buffer2s_mmm,
//      B1_offset_list_mm_size, B1_offset_list_mm,
//      B2_offset_list_mm_size, B2_offset_list_mm,
//      B_block_rows_list_mm_size, B_block_rows_list_mm,
//      B_block_cols_list_mm_size, B_block_cols_list_mm,
//      B_buffer_readys_mm_size, B_buffer_readys_mm,
//      compute_worker_start,
//      num_local_compute_workers,
//      compute_workers_finished_mm_size, compute_workers_finished_mm,
//      num_compute_workers);
//  bg_thread.detach();
  _bg_aux_worker_pull(
      A_m_size, A_m,
      A1,
      A2,
      A1_tile,
      A2_tile,
      A_buffer1s_mmm_size, A_buffer1s_mmm,
      A_buffer2s_mmm_size, A_buffer2s_mmm,
      A1_offset_list_mm_size, A1_offset_list_mm,
      A2_offset_list_mm_size, A2_offset_list_mm,
      A_block_rows_list_mm_size, A_block_rows_list_mm,
      A_block_cols_list_mm_size, A_block_cols_list_mm,
      A_buffer_readys_mm_size, A_buffer_readys_mm,
      B_m_size, B_m,
      B1,
      B2,
      B1_tile,
      B2_tile,
      B_buffer1s_mmm_size, B_buffer1s_mmm,
      B_buffer2s_mmm_size, B_buffer2s_mmm,
      B1_offset_list_mm_size, B1_offset_list_mm,
      B2_offset_list_mm_size, B2_offset_list_mm,
      B_block_rows_list_mm_size, B_block_rows_list_mm,
      B_block_cols_list_mm_size, B_block_cols_list_mm,
      B_buffer_readys_mm_size, B_buffer_readys_mm,
      compute_worker_start,
      num_local_compute_workers,
      compute_workers_finished_mm_size, compute_workers_finished_mm,
      num_compute_workers);
}
void _bg_aux_worker_pull(
    /* double *A */                                       int64_t A_m_size, void *A_m,
    /* uint64_t A1 */                                     uint64_t A1,
    /* uint64_t A2 */                                     uint64_t A2,
    /* uint64_t A1_tile */                                uint64_t A1_tile,
    /* uint64_t A2_tile */                                uint64_t A2_tile,
    /* std::vector<double *> &A_buffer1s */               int64_t A_buffer1s_mmm_size, void *A_buffer1s_mmm,
    /* std::vector<double *> &A_buffer2s */               int64_t A_buffer2s_mmm_size, void *A_buffer2s_mmm,
    /* std::vector<uint64_t> &A1_offset_list */           int64_t A1_offset_list_mm_size, void *A1_offset_list_mm,
    /* std::vector<uint64_t> &A2_offset_list */           int64_t A2_offset_list_mm_size, void *A2_offset_list_mm,
    /* std::vector<uint64_t> &A_block_rows_list */        int64_t A_block_rows_list_mm_size, void *A_block_rows_list_mm,
    /* std::vector<uint64_t> &A_block_cols_list */        int64_t A_block_cols_list_mm_size, void *A_block_cols_list_mm,
    /* std::vector<int8_t *> &A_buffer_readys */          int64_t A_buffer_readys_mm_size, void *A_buffer_readys_mm,
    /* double *B */                                       int64_t B_m_size, void *B_m,
    /* uint64_t B1 */                                     uint64_t B1,
    /* uint64_t B2 */                                     uint64_t B2,
    /* uint64_t B1_tile */                                uint64_t B1_tile,
    /* uint64_t B2_tile */                                uint64_t B2_tile,
    /* std::vector<double *> &B_buffer1s */               int64_t B_buffer1s_mmm_size, void *B_buffer1s_mmm,
    /* std::vector<double *> &B_buffer2s */               int64_t B_buffer2s_mmm_size, void *B_buffer2s_mmm,
    /* std::vector<uint64_t> &B1_offset_list */           int64_t B1_offset_list_mm_size, void *B1_offset_list_mm,
    /* std::vector<uint64_t> &B2_offset_list */           int64_t B2_offset_list_mm_size, void *B2_offset_list_mm,
    /* std::vector<uint64_t> &B_block_rows_list */        int64_t B_block_rows_list_mm_size, void *B_block_rows_list_mm,
    /* std::vector<uint64_t> &B_block_cols_list */        int64_t B_block_cols_list_mm_size, void *B_block_cols_list_mm,
    /* std::vector<int8_t *> &B_buffer_readys */          int64_t B_buffer_readys_mm_size, void *B_buffer_readys_mm,
    /* uint64_t compute_worker_start */                   uint64_t compute_worker_start,
    /* uint64_t num_local_compute_workers */              uint64_t num_local_compute_workers,
    /* std::vector<int8_t *> &compute_workers_finished */ int64_t compute_workers_finished_mm_size, void *compute_workers_finished_mm,
    /* uint64_t num_compute_workers */                    uint64_t num_compute_workers)
{
  double *A = _get_pointer_from_memref<double>(A_m_size, A_m);
  UnrankedMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer1s_mmm_unranked = {A_buffer1s_mmm_size, A_buffer1s_mmm};
  DynamicMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer1s_mmm_dynamic(A_buffer1s_mmm_unranked);
  UnrankedMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer2s_mmm_unranked = {A_buffer2s_mmm_size, A_buffer2s_mmm};
  DynamicMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> A_buffer2s_mmm_dynamic(A_buffer2s_mmm_unranked);
  std::vector<uint64_t *> A1_offset_list = _get_vector_of_pointers<uint64_t>(A1_offset_list_mm_size, A1_offset_list_mm, num_compute_workers);
  std::vector<uint64_t *> A2_offset_list = _get_vector_of_pointers<uint64_t>(A2_offset_list_mm_size, A2_offset_list_mm, num_compute_workers);
  std::vector<uint64_t *> A_block_rows_list = _get_vector_of_pointers<uint64_t>(A_block_rows_list_mm_size, A_block_rows_list_mm, num_compute_workers);
  std::vector<uint64_t *> A_block_cols_list = _get_vector_of_pointers<uint64_t>(A_block_cols_list_mm_size, A_block_cols_list_mm, num_compute_workers);
  std::vector<int8_t *> A_buffer_readys = _get_vector_of_pointers<int8_t>(A_buffer_readys_mm_size, A_buffer_readys_mm, num_compute_workers);
  double *B = _get_pointer_from_memref<double>(B_m_size, B_m);
  UnrankedMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> B_buffer1s_mmm_unranked = {B_buffer1s_mmm_size, B_buffer1s_mmm};
  DynamicMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> B_buffer1s_mmm_dynamic(B_buffer1s_mmm_unranked);
  UnrankedMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> B_buffer2s_mmm_unranked = {B_buffer2s_mmm_size, B_buffer2s_mmm};
  DynamicMemRefType<UnrankedMemRefType<UnrankedMemRefType<double>>> B_buffer2s_mmm_dynamic(B_buffer2s_mmm_unranked);
  std::vector<uint64_t *> B1_offset_list = _get_vector_of_pointers<uint64_t>(B1_offset_list_mm_size, B1_offset_list_mm, num_compute_workers);
  std::vector<uint64_t *> B2_offset_list = _get_vector_of_pointers<uint64_t>(B2_offset_list_mm_size, B2_offset_list_mm, num_compute_workers);
  std::vector<uint64_t *> B_block_rows_list = _get_vector_of_pointers<uint64_t>(B_block_rows_list_mm_size, B_block_rows_list_mm, num_compute_workers);
  std::vector<uint64_t *> B_block_cols_list = _get_vector_of_pointers<uint64_t>(B_block_cols_list_mm_size, B_block_cols_list_mm, num_compute_workers);
  std::vector<int8_t *> B_buffer_readys = _get_vector_of_pointers<int8_t>(B_buffer_readys_mm_size, B_buffer_readys_mm, num_compute_workers);
  std::vector<int8_t *> compute_workers_finished = _get_vector_of_pointers<int8_t>(compute_workers_finished_mm_size, compute_workers_finished_mm, num_compute_workers);


  uint64_t compute_worker_bound = compute_worker_start + num_local_compute_workers;
  while (std::any_of(compute_workers_finished.begin() + compute_worker_start,
                     compute_workers_finished.begin() + compute_worker_bound,
                     [](int8_t *is_finished) -> bool {
                       return !_atomic_load_n_i8(is_finished);
  })) {
    for (uint64_t w = compute_worker_start; w < compute_worker_bound; ++w) {
//      if (A_buffer_readys[w]->load(std::memory_order_acquire)) {
      if (_atomic_load_n_i8(A_buffer_readys[w])) {
        continue;
      }
      uint64_t A1_offset = *A1_offset_list[w];
      uint64_t A2_offset = *A2_offset_list[w];
      uint64_t A_block_rows = *A_block_rows_list[w];
      uint64_t A_block_cols = *A_block_cols_list[w];
//      double **A_buffer1 = &A_buffer1s[w];
//      double **A_buffer2 = &A_buffer2s[w];
      UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1s_mm_unranked = A_buffer1s_mmm_dynamic.data[w];
      DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1s_mm_dynamic(A_buffer1s_mm_unranked);
      UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer2s_mm_unranked = A_buffer2s_mmm_dynamic.data[w];
      DynamicMemRefType<UnrankedMemRefType<double>> A_buffer2s_mm_dynamic(A_buffer2s_mm_unranked);
      DynamicMemRefType<double> A_buffer2s_m_dynamic(A_buffer2s_mm_dynamic.data[0]);
      double *A_buffer2 = A_buffer2s_m_dynamic.data;
      _copy_to_buffer(A,
                      A1, A2,
                      A1_offset, A2_offset,
                      A_buffer2,
                      A1_tile, A2_tile,
                      A_block_rows, A_block_cols);
      _swap_buffer(A_buffer1s_mm_dynamic, A_buffer2s_mm_dynamic);
//      /// test
//      {
//        if (A1_offset == 4) {
//          std::lock_guard<std::mutex> lock(log_mutex);
//          std::cout << "!!!aux_worker on A_buffer2:" << std::endl;
//          _print_matrix_buffer(A_buffer2, A1_tile, A2_tile, A_block_rows, A_block_cols);
//          std::cout << "!!!aux_worker from A:" << std::endl;
//          _print_matrix_buffer(A, A1, A2, A1, A2);
//        }
//      }
//      /// end test
//      A_buffer_readys[w]->store(true, std::memory_order_release);
      _atomic_store_n_i8(A_buffer_readys[w], (int8_t) 1);
    }

    for (uint64_t w = compute_worker_start; w < compute_worker_bound; ++w) {
//      if (B_buffer_readys[w]->load(std::memory_order_acquire)) {
      if (_atomic_load_n_i8(B_buffer_readys[w])) {
        continue;
      }
      uint64_t B1_offset = *B1_offset_list[w];
      uint64_t B2_offset = *B2_offset_list[w];
      uint64_t B_block_rows = *B_block_rows_list[w];
      uint64_t B_block_cols = *B_block_cols_list[w];
//      double **B_buffer1 = &B_buffer1s[w];
//      double **B_buffer2 = &B_buffer2s[w];
      UnrankedMemRefType<UnrankedMemRefType<double>> B_buffer1s_mm_unranked = B_buffer1s_mmm_dynamic.data[w];
      DynamicMemRefType<UnrankedMemRefType<double>> B_buffer1s_mm_dynamic(B_buffer1s_mm_unranked);
      UnrankedMemRefType<UnrankedMemRefType<double>> B_buffer2s_mm_unranked = B_buffer2s_mmm_dynamic.data[w];
      DynamicMemRefType<UnrankedMemRefType<double>> B_buffer2s_mm_dynamic(B_buffer2s_mm_unranked);
      DynamicMemRefType<double> B_buffer2s_m_dynamic(B_buffer2s_mm_dynamic.data[0]);
      double *B_buffer2 = B_buffer2s_m_dynamic.data;
      _copy_to_buffer(B,
                      B1, B2,
                      B1_offset, B2_offset,
                      B_buffer2,
                      B1_tile, B2_tile,
                      B_block_rows, B_block_cols);
      _swap_buffer(B_buffer1s_mm_dynamic, B_buffer2s_mm_dynamic);
//      B_buffer_readys[w]->store(true, std::memory_order_release);
      _atomic_store_n_i8(B_buffer_readys[w], (int8_t) 1);
    }
  }
}

void _do_gemm_on_buffer(
    DynamicMemRefType<UnrankedMemRefType<double>> &A_buffer_active_mm_dynamic, /* double *A_buffer_active */
    uint64_t A1_tile, uint64_t A2_tile,
    uint64_t A_block_rows, uint64_t A_block_cols,
    DynamicMemRefType<UnrankedMemRefType<double>> &B_buffer_active_mm_dynamic, /* double *B_buffer_active */
    uint64_t B1_tile, uint64_t B2_tile,
    uint64_t B_block_rows, uint64_t B_block_cols,
    double *C,
    uint64_t C1, uint64_t C2,
    uint64_t C1_offset, uint64_t C2_offset)
{
  UnrankedMemRefType<double> A_buffer_active_unranked = A_buffer_active_mm_dynamic.data[0];
  DynamicMemRefType<double> A_buffer_active_dynamic(A_buffer_active_unranked);
  double *A_buffer_active = A_buffer_active_dynamic.data;
  UnrankedMemRefType<double> B_buffer_active_unranked = B_buffer_active_mm_dynamic.data[0];
  DynamicMemRefType<double> B_buffer_active_dynamic(B_buffer_active_unranked);
  double *B_buffer_active = B_buffer_active_dynamic.data;

//  /// test
//  {
//    std::lock_guard<std::mutex> lock(log_mutex);
//    std::cout << "A_buffer_active" << std::endl;
//    _print_matrix_buffer(A_buffer_active, A1_tile, A2_tile);
//    std::cout << "B_buffer_active" << std::endl;
//    _print_matrix_buffer(B_buffer_active, B1_tile, B2_tile);
//  }
//  /// end test

  for (uint64_t i = 0; i < A_block_rows; ++i) {
    uint64_t c_i = C1_offset + i;
    for (uint64_t k = 0; k < A_block_cols; ++k) {
      for (uint64_t j = 0; j < B_block_cols; ++j) {
        uint64_t c_j = C2_offset + j;
        C[c_i * C2 + c_j] += A_buffer_active[i * A2_tile + k] * B_buffer_active[k * B2_tile + j];
      }
    }
  }

//  /// test
//  {
//
//    for (uint64_t i = 0; i < A_block_rows; ++i) {
//      uint64_t c_i = C1_offset + i;
//      if (c_i == 4) {
//        for (uint64_t j = 0; j < B_block_cols; ++j) {
//          uint64_t c_j = C2_offset + j;
//          std::lock_guard<std::mutex> lock(log_mutex);
//          std::cout << "C[" << c_i << ", " << c_j << "]: " << C[c_i * C2 + c_j] << std::endl;
//        }
//        {
//          std::lock_guard<std::mutex> lock(log_mutex);
//          std::cout << "A_buffer_active:" << std::endl;
//          _print_matrix_buffer(A_buffer_active, A1_tile, A2_tile, A_block_rows, A_block_cols);
//          std::cout << "B_buffer_active:" << std::endl;
//          _print_matrix_buffer(B_buffer_active, B1_tile, B2_tile, B_block_rows, B_block_cols);
//        }
//      }
//    }
//  }
//  /// end test
}