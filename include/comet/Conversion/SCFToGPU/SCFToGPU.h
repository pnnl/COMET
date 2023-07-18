//===- SCFToGPU.h - Convert SCF dialect to GPU dialect --*- C++ -*-===//
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

#ifndef COMET_CONVERSION_SCFTOGPU_H
#define COMET_CONVERSION_SCFTOGPU_H

#include "mlir/Support/LLVM.h"

namespace mlir
{
    class Pass;
    class RewritePatternSet;

    namespace comet
    {
#define GEN_PASS_DECL_CONVERTSCFTOGPU
#include "comet/Conversion/Passes.h.inc"

        /// Collect a set of patterns to convert TensorAlgebra operations to SCF dialect.
        /// Subsequently lower the SCF dialect to GPU dialect
        void populateSCFToGPUConversionPatterns(RewritePatternSet &patterns);

        /// Lowers TensorAlgebra operations
        std::unique_ptr<Pass> createLowerSCFToGPUPass();

        /// Convert Linalg ops to Vector.
        void populateGPUVectorizationPatterns(RewritePatternSet &patterns);

        std::unique_ptr<Pass> createGPUVectorizationPass(int64_t maxVectorSize = 4096);

        void populateGPUReduceSharedMemoryBankConflictsPatterns(RewritePatternSet &patterns);

        /// Apply transformation to reduce the number of bank conflicts when accessing
        /// shared memory by padding fastest moving dimension with the specified size.
        std::unique_ptr<Pass> createGPUReduceSharedMemoryBankConflictsPass(int64_t paddingSizeBits = 128);

        /// Various pipelining strategies
        /// Pipeline shared memory copy by apply software pipelining scheduling where
        /// copy to shared memory is in stage 0 and the rest of the operations are in
        /// stage `depth - 1`.
        enum class PipeliningSchedulingStrategy {
         // Schedule the load from global memory into stage 0 and the associated store
         // will be in stage depth - 1.
         loadGlobalStage0 = 0,
         // Schedule both the load from global and the store to shared memory in stage
         // 0. The compute operations will be in stage depth-1. This means there won't
         // be vector registers carried between stages.
         loadStoreStage0 = 1,
         // Schedule optimized when using nvidia tensorcore with async copies. It will
         // set all the copies in stage 0 then it will prefecth part of loads in `depth
         // - 2` stage and keep the rest of the load and compute into `depth - 1`.
         nvidiaTensorCore = 2,
        };

        /// *** perform GPU S/W pipelining.
        void populateGPUPipeliningPatterns(RewritePatternSet &patterns);

        std::unique_ptr<Pass> createGPUPipeliningPass(bool epiloguePeeling = true, unsigned depth = 1,
                        PipeliningSchedulingStrategy schedule = PipeliningSchedulingStrategy::loadGlobalStage0);

        /// Pass to optimize vector transfer_read and transfer_write.
        void populateGPUOptimizeVectorTransferPatterns(RewritePatternSet &patterns);

        std::unique_ptr<Pass> createGPUOptimizeVectorTransferPass(bool flatten = false);

        //*** affine-super-vectorize
        std::unique_ptr<Pass> createAffineVectorizePass();

        //*** creates gpu.host_register ops
        std::unique_ptr<Pass> createGPUHostRegisterOpPass();  

        //*** create memref.copy ops
        std::unique_ptr<Pass> createGPUMemrefCopyPass();      
    }
} // namespace mlir

#endif // COMET_CONVERSION_SCFTOGPU_H
