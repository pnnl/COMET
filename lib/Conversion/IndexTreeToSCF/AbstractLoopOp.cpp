//===- AbstractLoopOp.h - A helper class AbstractLoopOp to lower Index Tree dialect to SCF dialect --*- C++ -*-===//
//
// Copyright 2024 Battelle Memorial Institute
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
// This file implements a helper class AbstractLoopOp used in the lowering of
// index tree dialect to SCF dialect. The class abstract several SCF loop
// operations, such as scf::ForOp, scf::ParallelOp.
//===----------------------------------------------------------------------===//

#include "comet/Conversion/IndexTreeToSCF/AbstractLoopOp.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"

std::unordered_set<std::string> AbstractLoopOp::supported_types = {"default",
                                                                   "serial",
                                                                   "parallel",
                                                                   "omp.parallel",
                                                                   "reduction",
                                                                   "window"};

void AbstractLoopOp::setIteratorType(std::string type)
{
  if (supported_types.find(type) != supported_types.end())
  {
    iteratorType = type;
  }
  else
  {
    llvm::errs() << "Unsupported iterator type " + type + "\n";
  }
}

void AbstractLoopOp::setOp(scf::ForOp forOp, std::string iterator_type)
{
  setIteratorType(iterator_type);
  op = forOp;
}

void AbstractLoopOp::setOp(scf::ParallelOp parallelOp, std::string iterator_type)
{
  setIteratorType(iterator_type);
  op = parallelOp;
}

void AbstractLoopOp::setOp(omp::WsloopOp parallelOp, std::string iterator_type)
{
  setIteratorType(iterator_type);
  op = parallelOp;
}

void AbstractLoopOp::setLowerBound(mlir::Value &lowerBound)
{
  if (iteratorType == "parallel")
  {
    llvm::errs() << "Error: scf::ParallelOp does not support setLowerBound.\n";
  }
  else if ("omp.parallel" == iteratorType)
  {
    llvm::errs() << "Error: omp::WsloopOp does not support setLowerBound.\n";
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    handle.setLowerBound(lowerBound);
  }
}

mlir::Value AbstractLoopOp::getUpperBound()
{
  if (iteratorType == "parallel")
  {
    auto handle = mlir::dyn_cast<scf::ParallelOp>(op);
    return handle.getUpperBound()[0];
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    return handle.getUpperBound();
  }
}

mlir::Value AbstractLoopOp::getLowerBound()
{
  if (iteratorType == "parallel")
  {
    auto handle = mlir::dyn_cast<scf::ParallelOp>(op);
    return handle.getLowerBound()[0];
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    return handle.getLowerBound();
  }
}

void AbstractLoopOp::setUpperBound(mlir::Value &upperBound)
{
  if (iteratorType == "parallel")
  {
    llvm::errs() << "Error: scf::ParallelOp does not support setUpperBound.\n";
  }
  else if ("omp.parallel" == iteratorType)
  {
    llvm::errs() << "Error: omp::WsloopOp does not support setUpperBound.\n";
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    handle.setUpperBound(upperBound);
  }
}

mlir::Block *AbstractLoopOp::getBody()
{
  if (iteratorType == "parallel")
  {
    auto handle = mlir::dyn_cast<scf::ParallelOp>(op);
    return handle.getBody();
  }
  else if ("omp.parallel" == iteratorType)
  {
    auto handle =  mlir::dyn_cast<omp::LoopNestOp>(mlir::dyn_cast<omp::WsloopOp>(op).getWrappedLoop());
    return &handle.getRegion().getBlocks().front();
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    return handle.getBody();
  }
}

mlir::Value AbstractLoopOp::getInductionVar()
{
  if (iteratorType == "parallel")
  {
    auto handle = mlir::dyn_cast<scf::ParallelOp>(op);
    return handle.getInductionVars()[0];
  }
  else if ("omp.parallel" == iteratorType)
  {
    auto handle =  mlir::dyn_cast<omp::LoopNestOp>(mlir::dyn_cast<omp::WsloopOp>(op).getWrappedLoop());
    return handle.getRegion().getArguments().front();
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    return handle.getInductionVar();
  }
}

void AbstractLoopOp::buildLoopOp(const std::string &type,
                                 mlir::OpBuilder &builder,
                                 const mlir::Location &loc,
                                 const mlir::Value &lowerBound,
                                 const mlir::Value &upperBound,
                                 const mlir::Value &step)
{
  setIteratorType(type);
  if ("parallel" == type)
  {
    // Is parallel for loop
    auto tmp = builder.create<scf::ParallelOp>(loc, lowerBound, upperBound, step);
    setOp(tmp, type);
  }
  else if ("omp.parallel" == type)
  {
    OpBuilder::InsertionGuard guard(builder);  /// RAII guard to reset the insertion point of the builder when destroyed.
    auto wsloop = builder.create<omp::WsloopOp>(loc); 
    builder.createBlock(&wsloop.getRegion(), {}, {}, {});
    auto tmp = builder.create<omp::LoopNestOp>(loc,
                                             ValueRange{lowerBound},
                                             ValueRange{upperBound},
                                             ValueRange{step});
    builder.createBlock(&tmp.getRegion(), {}, {builder.getIndexType()}, {tmp.getLoc()});  /// NOTE: this changed the insertion point automatically.
    omp_wsloop_yield_op = builder.create<omp::YieldOp>(loc, ValueRange());
    builder.setInsertionPointToEnd(wsloop.getBody());
    builder.create<omp::TerminatorOp>(loc);
    setOp(wsloop, type);
  }
  else
  {
    // Is default for loop
    auto tmp = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    setOp(tmp, type);
  }
}

void AbstractLoopOp::dump()
{
  if (iteratorType == "parallel")
  {
    auto handle = mlir::dyn_cast<scf::ParallelOp>(op);
    handle.dump();
  }
  else if ("omp.parallel" == iteratorType)
  {
    auto handle = mlir::dyn_cast<omp::WsloopOp>(op);
    handle.dump();
  }
  else
  {
    auto handle = mlir::dyn_cast<scf::ForOp>(op);
    handle.dump();
  }
}