//===- Passes.h - Passes Definition implemented in Index Tree-----------------------------------===//
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
// This file exposes the entry points to create compiler passes for Index Tree dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INDEXTREE_PASSES_H
#define INDEXTREE_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir
{
  class Pass;

  namespace IndexTree
  {
    std::unique_ptr<mlir::Pass> createIndexTreePass();

    /// Create a pass for applying compressed workspace transformation into IndexTreeIR
    std::unique_ptr<Pass> createCompressedWorkspaceTransformsPass();

    /// Create a pass for lowering IndexTree IR ops to scf dialect version
    std::unique_ptr<Pass> createLowerIndexTreeIRToSCFPass();

    /// Create a pass for the redundancy-aware kernel fusion on index tree dialect for some compound expressions
    std::unique_ptr<Pass> createKernelFusionPass();
  } // end namespace IndexTree
} // end namespace mlir

#endif // INDEXTREE_PASSES_H
