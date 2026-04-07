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
void comet_print_matrix_f64(int64_t A_memref_size, void *A_memref, int64_t A1, int64_t A2);
int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref);
void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value);
void comet_swap_buffers(int64_t memref1_size, void *memref1, int64_t memref2_size, void *memref2);

void comet_test_array_of_buffers(
    uint64_t num_compute_workers,
    uint64_t A1_tile, uint64_t A2_tile,
    int64_t A_buffer1s_size, void *A_buffer1s);
void comet_test_write_to_buffer(
    uint64_t val,
    uint64_t A1_tile, uint64_t A2_tile,
    int64_t A_buffer_size, void *A_buffer);

/// -------------------------------------------------------------------------------
/// Parallel Double Buffering: multiple compute workers and less auxiliary workers
/// -------------------------------------------------------------------------------
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
    /* int8_t *is_finished */        int64_t is_finished_m_size, void *is_finished_m);
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
    /* uint64_t num_compute_workers */                    uint64_t num_compute_workers);

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
void _do_gemm_on_buffer(
    DynamicMemRefType<UnrankedMemRefType<double>> &A_buffer_active_mm_dynamic, /* double *A_buffer_active */
    uint64_t A1_tile, uint64_t A2_tile,
    uint64_t A_block_rows, uint64_t A_block_cols,
    DynamicMemRefType<UnrankedMemRefType<double>> &B_buffer_active_mm_dynamic, /* double *B_buffer_active */
    uint64_t B1_tile, uint64_t B2_tile,
    uint64_t B_block_rows, uint64_t B_block_cols,
    double *C,
    uint64_t C1, uint64_t C2,
    uint64_t C1_offset, uint64_t C2_offset);
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
    /* int8_t *is_finished */        int64_t is_finished_m_size, void *is_finished_m);
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
    /* uint64_t num_compute_workers */                    uint64_t num_compute_workers);


template <typename T>
std::vector<T *> _get_vector_of_pointers(int64_t mm_size,
                                         void *mm,
                                         uint64_t length)
{
  UnrankedMemRefType<UnrankedMemRefType<T>> mm_unranked = {mm_size, mm};
  DynamicMemRefType<UnrankedMemRefType<T>> mm_dynamic(mm_unranked);

  std::vector<T *> res(length);
  for (uint64_t i = 0; i < length; ++i) {
    UnrankedMemRefType<T> m_unranked = mm_dynamic.data[i];
    DynamicMemRefType<T> m_dynamic(m_unranked);
    res[i] = m_dynamic.data;
  }

  return res;
}

template <typename T>
T *_get_pointer_from_memref(int64_t memref_size, void *memref)
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

template <typename T>
void _print_matrix_buffer(T *buffer,
                          uint64_t A1, uint64_t A2,
                          uint64_t rows, uint64_t cols)
{
  std::cout << "_print_matrix_buffer [" << rows << "x" << cols << "] / [" << A1 << "x" << A2 << "]" << std::endl;
  for (uint64_t i = 0; i < rows; ++i) {
    for (uint64_t j = 0; j < cols; ++j) {
      std::cout << buffer[i * A2 + j] << " ";
    }
    std::cout << std::endl;
  }
}


#endif //COMET_DOUBLEBUFFERUTILS_H