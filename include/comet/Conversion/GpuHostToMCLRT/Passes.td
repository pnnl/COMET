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

#ifndef GPUHOST_TO_MCLRT_CONVERSION_PASSES
#define GPUHOST_TO_MCLRT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertGpuHostToMCLRTPass: Pass<"convert-gpu-host-to-mcl-rt", "mlir::ModuleOp"> {
    let summary = "Convert Gpu Host to MCR Runtime Calls";
    let description = [{

    }];
    let constructor = "mlir::comet::createConvertGpuHostToMCLRTPass()";

    let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
                             "mlir::spirv::SPIRVDialect"];

    let options = [
       Option<"xclbin_path", "xclbin-path",
              "const char*", /*default*/"\"-\"",
              "Path to xclbin">,
   ];
}
#endif
