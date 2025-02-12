//===- Patterns.h - Conversion Pass Construction and Registration -----------===//
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

#ifndef COMET_DIALECT_INDEXTREE_PATTERNS_H
#define COMET_DIALECT_INDEXTREE_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

#include "comet/Dialect/IndexTree/Analysis/CopiedDomainAnalysis.h"

namespace mlir
{
    namespace indexTree
    {
        void populateDomainInferencePatterns(MLIRContext *context, RewritePatternSet &patterns, CopiedDomainAnalysis &copiedDomains);
        void populateDomainConcretizationPatterns(MLIRContext *context, RewritePatternSet &patterns);
        void populateIndexTreeTypeConversionPatterns(MLIRContext *context, RewritePatternSet &patterns, TypeConverter &typeConverter, ConversionTarget& target);
        void populateIndexTreeInliningPatterns(MLIRContext *context, RewritePatternSet &patterns);
    }
}

#endif // COMET_DIALECT_INDEXTREE_PASSES_H
