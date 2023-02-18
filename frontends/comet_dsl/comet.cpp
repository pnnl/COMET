//===- comet.cpp - The COMET Compiler ----===//
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
// This file implements the entry point for the COMET compiler.
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/ITDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "MLIRGen.h"
#include "Parser.h"

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "comet_dsl"
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tensorAlgebra file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace
{
  enum InputType
  {
    TensorAlgebra,
    MLIR
  };
}

static cl::opt<enum InputType> inputType(
    "x", cl::init(TensorAlgebra), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(TensorAlgebra, "ta", "load the input file as a TensorAlgebra source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

// List of TA compiler parameters

// =============================================================================
// Emit AST or Tensor Algebra dialect
// =============================================================================
static cl::opt<bool> emitAST("emit-ast", cl::desc("Output the AST dump"));
static cl::opt<bool> emitTA("emit-ta", cl::desc("output the Tensor Algebra dialect dump"));
static cl::opt<bool> emitIT("emit-it", cl::desc("output the Index Tree dialect dump"));

// =============================================================================
// Optimization at the TA dialect (High-level optimizations) for tensor contraction operations
// =============================================================================
static cl::opt<bool> OptMultiOpFactorization("opt-multiop-factorize",
                                             cl::desc("Multi operations factorization optimization"));

static cl::opt<bool> IsSelectBestPermTTGT("opt-bestperm-ttgt",
                                          cl::desc("Select the best index permutation for TTGT, otherwise the first appropriate permutation"));

static cl::opt<int> selectedPermNum("perm-num", cl::init(1),
                                    cl::ZeroOrMore, cl::desc("Select the permutation number to choose"));

// =============================================================================
// Operation based optimizations
// =============================================================================
static cl::opt<bool> OptMatmulTiling("opt-matmul-tiling",
                                     cl::desc("Optimize LinAlg matmul operation with tiling"));

static cl::opt<bool> OptCallToMatMulMicroKernel("opt-matmul-mkernel",
                                                cl::desc("Replace the inner linalg.matmul that introduced after tiling with the blis micro kernel"));

static cl::opt<bool> OptDenseTransposeOp("opt-dense-transpose",
                                         cl::desc("Optimize transpose operation: optimal loop ordering and tiling"));

// =============================================================================
// Sparse kernel optimizations
// =============================================================================
static cl::opt<bool> OptWorkspace("opt-comp-workspace", cl::init(false),
                                  cl::desc("Optimize sparse output code generation while reducing iteration space for nonzero elements"));

// The details of the fusion algorithm can be found in the following paper.
// ReACT: Redundancy-Aware Code Generation for Tensor Expressions.
// Tong Zhou, Ruiqin Tian, Rizwan A Ashraf, Roberto Gioiosa, Gokcen Kestor, Vivek Sarkar.
// 2022 31st International Conference on Parallel Architectures and Compilation Techniques (PACT). October 2022.
//  =============================================================================
//  Partial Fusion (ReACT Fusion) on Index Tree dialect
//  =============================================================================
static cl::opt<bool> OptKernelFusion("opt-fusion", cl::init(false),
                                     cl::desc("Output IT dialect after redundancy-aware fusion"));

// =============================================================================
// TTGT reformulation for tensor contraction operations
// =============================================================================
static cl::opt<bool> IsLoweringTCtoTTGT("convert-tc-to-ttgt",
                                        cl::desc("Output IR after lowering dense tensor contractions operations through a TTGT approach"));

// =============================================================================
// Lowering TA operations to IT dialect
// =============================================================================
static cl::opt<bool> IsLoweringtoIndexTree("convert-ta-to-it", // Lower sparse/dense mult (semiring) and elemwise (monoid) ops to index-tree dialect
                                           cl::desc("Output IT dialect after processing dense sparse/dense mult and elemwise ops"));

// =============================================================================
// Lowering IT operations to loops
// =============================================================================
static cl::opt<bool> IsLoweringtoSCF("convert-to-loops", // Lower sparse/dense mult (semiring) and elemwise (monoid) ops to index-tree dialect
                                     cl::desc("Output IR after processing all ops"));

// =============================================================================
// Utility functions
// =============================================================================
static cl::opt<bool> IsPrintFlops("print-flops", cl::init(false),
                                  cl::desc("Print the flops per tensor contraction"));

/// Returns a Tensor Algebra AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<tensorAlgebra::ModuleAST> parseInputFile(llvm::StringRef filename)
{
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError())
  {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  // LexerBuffer lexer(buffer.begin(), buffer.end(), filename);
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{
  // Handle '.ta' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir"))
  {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError())
  {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module)
  {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningModuleRef &module)
{
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(mlir::tensorAlgebra::createRemoveLabeledTensorOpsPass());

  // Check to see if we are dumping to TA dialect.
  if (emitTA)
  {
    if (mlir::failed(pm.run(*module)))
      return 4;
    return 0;
  }

  //  =============================================================================
  //  High-level optimization at the TA dialect
  //  Such as finding the optimal ordering of dense tensor contractions, or reformulating tensor contractions
  //  operations via TTGT
  //  =============================================================================
  if (OptMultiOpFactorization)
  {
    /// createFindOptimalTCFactorizationPass should be before lowering of input/output tensor declarations
    /// because this pass finds the optimal ordering of dense tensor multiplication
    /// operations before lowering them specific tc operations
    optPM.addPass(mlir::tensorAlgebra::createFindOptimalTCFactorizationPass());
  }

  optPM.addPass(mlir::tensorAlgebra::createLowerTAMulChainPass()); // Lowering for chain operations
  //  =============================================================================

  //  =============================================================================
  //  Creating tensor declarations for temporal tensors in compound expressions, preprocessing.
  //  =============================================================================
  optPM.addPass(mlir::tensorAlgebra::createPreLoweringPass()); // Creating tensor declarations for temporal tensors in chain operations
  //  =============================================================================

  // ===================================================================================
  // Lowering of TC (tensor contraction) operation to Index Tree dialect
  // Also performs optimization at the Index Tree dialect
  // ===================================================================================
  if (IsLoweringtoIndexTree || emitIT)
  {
    /// Generate the index tree IR
    optPM.addPass(mlir::IndexTree::createIndexTreePass());

//    // Dump index tree dialect.
//    if (emitIT)
//    {
//      if (mlir::failed(pm.run(*module)))
//        return 4;
//      return 0;
//    }
  }

  if (OptKernelFusion)
  {
    // Apply partial fusion on index tree dialect for some compound expressions.
    optPM.addPass(mlir::IndexTree::createKernelFusionPass());
  }

  if (OptWorkspace)
  {
    // Optimized workspace transformations, reduce iteration space for nonzero elements
    optPM.addPass(mlir::IndexTree::createCompressedWorkspaceTransformsPass());
  }

  /// Added by Zhen Peng on 01/23/2023
  // Dump index tree dialect.
  if (emitIT)
  {
    if (mlir::failed(pm.run(*module)))
      return 4;
    return 0;
  }
  // =============================================================================

  // =============================================================================
  // Early lowering for the following passes to lower IndexTree and TA dialect to loops
  // Early lowering pases based on the sparsity property of inputs/outputs
  // =============================================================================

  /// Sparse input tensor declararion should be lowered before dense input tensor declaration
  // sparse input tensor declaration lowering, also generate sparse_output_tensor declaration if needed
  // input and output sparse tensor declaration lowering are distant and need different information
  optPM.addPass(mlir::tensorAlgebra::createSparseTensorDeclLoweringPass());
  optPM.addPass(mlir::tensorAlgebra::createDenseTensorDeclLoweringPass()); // dense input tensor declaration lowering
  optPM.addPass(mlir::tensorAlgebra::createTensorFillLoweringPass());
  // =============================================================================

  // TTGT reformulation for dense tensor contraction operations
  if (IsLoweringTCtoTTGT)
  {
    // Sparse input and dense input/output tensor declarations needed be lowered before for TTGT pass
    optPM.addPass(mlir::tensorAlgebra::createLoweringTTGTPass(IsSelectBestPermTTGT, selectedPermNum, IsPrintFlops));
  }

  // =============================================================================
  // Operation based optimizations
  // =============================================================================
  if (OptDenseTransposeOp) // Optimize Dense Transpose operation
  {
    // If it is a transpose of dense tensor, the rewrites rules replaces ta.transpose with linalg.copy.
    optPM.addPass(mlir::tensorAlgebra::createTransposeLoweringPass());
    optPM.addPass(mlir::tensorAlgebra::createOptDenseTransposePass());
  }

  if (OptMatmulTiling)
  {
    optPM.addPass(mlir::tensorAlgebra::createLinAlgMatmulTilingPass());
  }

  if (OptCallToMatMulMicroKernel)
  {
    optPM.addPass(mlir::tensorAlgebra::createLinAlgMatmulMicroKernelPass());
  }

  // =============================================================================
  // Lowering all the operations to loops
  // =============================================================================
  if (IsLoweringtoSCF)
  {
    /// Workspace transformations will create new dense tensor declarations, so we need to call createDenseTensorDeclLoweringPass
    optPM.addPass(mlir::tensorAlgebra::createDenseTensorDeclLoweringPass());            // early lowering for dense input/output
    optPM.addPass(mlir::tensorAlgebra::createTempSparseOutputTensorDeclLoweringPass()); // early lowering for sparse output tensor declaration for temporaries
    optPM.addPass(mlir::tensorAlgebra::createSparseOutputTensorDeclLoweringPass());     // early lowering for sparse output
    
    // The partial Fusion pass might add new tensor.fill operations
    optPM.addPass(mlir::tensorAlgebra::createTensorFillLoweringPass());
    optPM.addPass(mlir::tensorAlgebra::createPCToLoopsLoweringPass());

    // =============================================================================
    // Lowering of other operations such as transpose, sum, etc. to SCF dialect
    // =============================================================================
    // If it is a transpose of dense tensor, the rewrites rules replaces ta.transpose with linalg.copy.
    // If it is a transpose of sparse tensor, it lowers the code to make a runtime call to specific sorting algorithm
    optPM.addPass(mlir::tensorAlgebra::createReduceOpLowerToSCFPass());
    optPM.addPass(mlir::tensorAlgebra::createTransposeLoweringPass());

    //Finally lowering index tree to SCF dialect
    optPM.addPass(mlir::IndexTree::createLowerIndexTreeIRToSCFPass());

    //  =============================================================================
  }

  // =============================================================================
  // Late lowering passes
  // =============================================================================
  optPM.addPass(mlir::tensorAlgebra::createSTCRemoveDeadOpsPass());
  optPM.addPass(mlir::tensorAlgebra::createLateLoweringPass());
  optPM.addPass(mlir::tensorAlgebra::createLowerLinAlgFillPass());
  optPM.addPass(mlir::createCSEPass());
  // =============================================================================

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST()
{
  if (inputType == InputType::MLIR)
  {
    llvm::errs() << "Can't dump a TensorAlgebra AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv)
{

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);

  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "Tensor Algebra compiler\n");

  if (emitAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  // Register our Dialect with MLIR.
  context.loadDialect<mlir::tensorAlgebra::TADialect>();
  context.loadDialect<mlir::indexTree::ITDialect>();
  context.loadDialect<mlir::StandardOpsDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();

  mlir::OwningModuleRef module;

  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  module->dump();
  return 0;
}