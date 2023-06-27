//===- blis_interface.cpp - Simple Blis subset interface -----------------===//
//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// Simple Blis subset interface implementation.
//
//===----------------------------------------------------------------------===//

#include "comet/ExecutionEngine/blis_interface.h"
#include "comet/ExecutionEngine/generic_mkernel.h"
#include "llvm/Support/raw_ostream.h"

#include <assert.h>
#include <iostream>


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
void bli_dgemm_x86_asm_6x8(
    dim_t m,
    dim_t n,
    dim_t k,
    double *restrict alpha,
    double *restrict a,
    double *restrict b,
    double *restrict beta,
    double *restrict c, inc_t rs_c0, inc_t cs_c0,
    auxinfo_t *restrict data,
    cntx_t *restrict cntx)
{
  // get the micro - arch
  arch_t id = bli_cpuid_query_id();
  const char *s = bli_arch_string(id);

  if ((strcmp("haswell", s) == 0) ||
      (strcmp("zen", s) == 0) || (strcmp("zen2", s) == 0) || (strcmp("zen3", s) == 0) ||
      (strcmp("skx", s) == 0) || (strcmp("knl", s) == 0))
  {
    bli_dgemm_haswell_asm_6x8(m, n, k, alpha, a, b, beta, c, rs_c0, cs_c0, data, cntx);
  }
  else
  {
    llvm::errs() << "Undefined microkernel"
                 << "\n";
  }
}
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)
void bli_dgemm_arm_asm_6x8(
    dim_t m,
    dim_t n,
    dim_t k,
    double *restrict alpha,
    double *restrict a,
    double *restrict b,
    double *restrict beta,
    double *restrict c, inc_t rs_c0, inc_t cs_c0,
    auxinfo_t *restrict data,
    cntx_t *restrict cntx)
{
  // get the micro - arch
  arch_t id = bli_cpuid_query_id();
  const char *s = bli_arch_string(id);

  // if ((strcmp("haswell", s) == 0) ||
  //     (strcmp("zen", s) == 0) || (strcmp("zen2", s) == 0) || (strcmp("zen3", s) == 0) ||
  //     (strcmp("skx", s) == 0) || (strcmp("knl", s) == 0))
  if (1)
  {
    bli_dgemm_armv8a_asm_6x8(m, n, k, alpha, a, b, beta, c, rs_c0, cs_c0, data, cntx);
  }
  else
  {
    llvm::errs() << "Undefined microkernel"
                 << "\n";
  }
}

#endif

extern "C" void _mlir_ciface_linalg_matmul_viewsxsxf64_viewsxsxf64_viewsxsxf64(
    StridedMemRefType<double, 2> *A, StridedMemRefType<double, 2> *B,
    StridedMemRefType<double, 2> *C)
{
  if (A->strides[1] != B->strides[1] || A->strides[1] != C->strides[1] ||
      A->strides[1] != 1 || A->sizes[0] < A->strides[1] ||
      B->sizes[0] < B->strides[1] || C->sizes[0] < C->strides[1] ||
      C->sizes[0] != A->sizes[0] || C->sizes[1] != B->sizes[1] ||
      A->sizes[1] != B->sizes[0])
  {
    printMemRefMetaData(std::cerr, *A);
    printMemRefMetaData(std::cerr, *B);
    printMemRefMetaData(std::cerr, *C);
    return;
  }

  double alpha = 1.0f;
  double beta = 1.0f;
  if (beta == -1.0)
  {
    alpha *= -1.0;
    beta = 1.0;
  }

  // get the micro-arch
  // arch_t id = bli_cpuid_query_id();
  // const char *s = bli_arch_string(id);

  // check the micro-arch and call the micro-kernel accordingly.
  // according to blis, the haswell dgemm micro-kernel can be executed on multiple micro-archs.
  // if ((strcmp("haswell", s) == 0) ||
  //     (strcmp("zen", s) == 0) || (strcmp("zen2", s) == 0) || (strcmp("zen3", s) == 0) ||
  //     (strcmp("skx", s) == 0) || (strcmp("knl", s) == 0))
  // {
  // bli_dgemm_haswell_asm_6x8
  bli_dgemm_asm_6x8(A->sizes[0], // m
                    B->sizes[1], // n
                    A->sizes[1], // k
                    &alpha,
                    A->data + A->offset,
                    B->data + B->offset,
                    &beta,
                    C->data + C->offset, C->strides[0], C->strides[1],
                    NULL, NULL);
  // }
  // else
  // {
  //   // printf("WARNING: falling back to a generic gemm implementation that is arch-independent.\n");
  //  dgemm_generic_noopt_mxn((int64_t)A->sizes[0], // m
  //                                                //                           (int64_t)B->sizes[1], // n
  //                                                //                           (int64_t)A->sizes[1], // k
  //                                                //                           &alpha,
  //                                                //                           A->data + A->offset,
  //                                                //                           B->data + B->offset,
  //                                                //                           &beta,
  //                                                //                           C->data + C->offset,
  //                                                //                           (int64_t)C->strides[0], (int64_t)C->strides[1]);
  //                                                // }
}


