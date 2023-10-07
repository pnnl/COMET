//===- blis_interface.h - Simple Blis subset interface -------------------===//
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
#ifndef COMET_BLIS_INTERFACE_H_
#define COMET_BLIS_INTERFACE_H_

#include "mlir/ExecutionEngine/RunnerUtils.h"

// suppress all warnings coming from inclusion of blis.h in source tree
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "blis.h"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.h"
#endif

#ifdef _WIN32
#ifndef COMET_BLIS_INTERFACE_EXPORT
#ifdef comet_blis_interface_EXPORTS
/* We are building this library */
#define COMET_BLIS_INTERFACE_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define COMET_BLIS_INTERFACE_EXPORT __declspec(dllimport)
#endif // comet_blis_interface_EXPORTS
#endif // COMET_BLIS_INTERFACE_EXPORT
#else
#define COMET_BLIS_INTERFACE_EXPORT
#endif // _WIN32

extern "C" COMET_BLIS_INTERFACE_EXPORT void
_mlir_ciface_linalg_matmul_viewsxsxf64_viewsxsxf64_viewsxsxf64(
    StridedMemRefType<double, 2> *A, StridedMemRefType<double, 2> *B,
    StridedMemRefType<double, 2> *C);

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // COMET_BLIS_INTERFACE_H_

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define bli_dgemm_asm_6x8 bli_dgemm_x86_ukr
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)
#define bli_dgemm_asm_6x8 bli_dgemm_arm_ukr
#else
#define bli_dgemm_asm_6x8 dgemm_generic_noopt_mxn
#endif
