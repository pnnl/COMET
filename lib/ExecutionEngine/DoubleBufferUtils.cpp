//
// Created by Zhen Peng on 10/3/25.
//

#include "comet/ExecutionEngine/DoubleBufferUtils.h"

#include <iostream>
#include <thread>
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
void _milr_ciface_comet_sanity_check(UnrankedMemRefType<UnrankedMemRefType<double>> *A_unranked)
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
  _milr_ciface_comet_sanity_check(&A_unranked);
}

int8_t comet_atomic_load_n_i8(int64_t memref_size, void *memref)
{
  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
  return __atomic_load_n(&flag_dynamic.data[0], __ATOMIC_ACQUIRE);
}

void comet_atomic_store_n_i8(int64_t memref_size, void *memref, int8_t value)
{
  UnrankedMemRefType<int8_t> flag_unranked = {memref_size, memref};
  DynamicMemRefType<int8_t> flag_dynamic(flag_unranked);
  __atomic_store_n(&flag_dynamic.data[0], value, __ATOMIC_RELEASE);
}

void comet_swap_buffers(int64_t memref1_size, void *memref1, int64_t memref2_size, void *memref2)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_unranked = {memref1_size, memref1};
  DynamicMemRefType<UnrankedMemRefType<double>> A_dynamic(A_unranked);

  UnrankedMemRefType<UnrankedMemRefType<double>> B_unranked = {memref2_size, memref2};
  DynamicMemRefType<UnrankedMemRefType<double>> B_dynamic(B_unranked);

  std::swap(A_dynamic.data[0], B_dynamic.data[0]);

}


void background_process(DynamicMemRefType<UnrankedMemRefType<double>> &A_buffer1_dynamic)
{
  DynamicMemRefType<double> A_buffer1_inside_dynamic(A_buffer1_dynamic.data[0]);
  for (int64_t i = 0; i < A_buffer1_inside_dynamic.sizes[0]; ++i) {
    A_buffer1_inside_dynamic.data[i] = i + 1;
  }
}

void comet_initialize_double_buffer_thread(int64_t A_memref1_size, void *A_memref1)
{
  UnrankedMemRefType<UnrankedMemRefType<double>> A_buffer1_unranked = {A_memref1_size, A_memref1};
  DynamicMemRefType<UnrankedMemRefType<double>> A_buffer1_dynamic(A_buffer1_unranked);

  std::thread background_thread(background_process, std::ref(A_buffer1_dynamic));
  background_thread.join();
}