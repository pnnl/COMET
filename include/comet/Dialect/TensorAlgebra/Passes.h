//===- TAPasses.h - Passes Definition implemented in Tensor Algebra-----------------------------------===//
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
// =============================================================================
//
// This file exposes the entry points to create compiler passes for TensorAlgebra.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORALGEBRA_PASSES_H
#define TENSORALGEBRA_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir
{
    class Pass;

    namespace tensorAlgebra
    {
        std::unique_ptr<Pass> createLinAlgMatmulTilingPass();

        // Optimize dense transpose (linalg.copy) based on the following paper:
        // HPTT: A High-Performance Tensor Transposition C++ Library
        // https://arxiv.org/abs/1704.04374
        std::unique_ptr<Pass> createOptDenseTransposePass(uint64_t tile_size = 1,
                                                          bool seperate_tiles = false);

        std::unique_ptr<Pass> createLinAlgMatmulMicroKernelPass();

        std::unique_ptr<Pass> createLowerLinAlgFillPass();

        std::unique_ptr<Pass> createFindOptimalTCFactorizationPass();

        std::unique_ptr<Pass> createLowerTAMulChainPass();

        /// Create a pass for pre lowering
        std::unique_ptr<Pass> createPreLoweringPass();

        /// Create a pass for lowering to the rest of the operations in `Std` dialects,
        /// such as printOp, constantOp, ReturnOp..
        std::unique_ptr<mlir::Pass> createLateLoweringPass();

        /// Create a pass for lowering TA operations to TTGT
        /// This pass selects either the best permutation among all
        /// or pass can specify the iteration order of the permutation, ith permutation
        std::unique_ptr<Pass> createLoweringTTGTPass(bool enableBestPerm,
                                                     int whatPermID = 1,
                                                     bool printFlops = false);

        /// Create a pass to lower dense input/output tensor declarations
        std::unique_ptr<Pass> createDenseTensorDeclLoweringPass();

        /// Create a pass to lower sparse input tensor declarations
        std::unique_ptr<Pass> createSparseTensorDeclLoweringPass();

        /// Create a pass for lowering sparse tensor output tensor decl operations
        std::unique_ptr<Pass> createSparseOutputTensorDeclLoweringPass();

        /// Create a pass for lowering temporal sparse tensor output tensor decl operations generated for compound expressions
        std::unique_ptr<Pass> createTempSparseOutputTensorDeclLoweringPass();

        /// Create a pass for lowering tensor fill operation to linalg.fill
        std::unique_ptr<Pass> createTensorFillLoweringPass();

        /// Create a pass for lowering Reduce operation to SCF
        // std::unique_ptr<Pass> createReduceOpLowerToSCFPass();

        /// Create a pass for lowering tensor operatations to other lower level dialects
        /// such as tensor elementwise addition, subtract, transpose, etc.
        std::unique_ptr<mlir::Pass> createTensorOpsLoweringPass();

        /// Create a pass for lowering transpose operations to SCF
        // std::unique_ptr<Pass> createTransposeLoweringPass();

        /// Create a pass for lowering sparse TA operations to SCFDimAttr
        std::unique_ptr<Pass> createSTCRemoveDeadOpsPass();

        /// Create a pass for lowering sparse TA operations to SCFDimAttrGPU
        std::unique_ptr<Pass> createSCFToSCFParallelPass();

        /// Create a pass for lowering programming constructs to SCF ops
        std::unique_ptr<Pass> createPCToLoopsLoweringPass();

        // TODO(gkestor): this pass is a workout to handle redundant LabeledTensor operations
        std::unique_ptr<Pass> createRemoveLabeledTensorOpsPass();

        std::unique_ptr<Pass> createFuncOpLoweringPass();

    } // end namespace tensorAlgebra
} // end namespace mlir

#endif // TENSORALGEBRA_PASSES_H
