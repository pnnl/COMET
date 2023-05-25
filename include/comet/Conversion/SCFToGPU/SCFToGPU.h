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
    }
} // namespace mlir

#endif // COMET_CONVERSION_SCFTOGPU_H
