#include "utils/dummyStats.h"

#include <cassert>
#include <iostream>

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace std;
using namespace dummyStats;

std::map<std::string, mlir::dummyStats::datum> mlir::dummyStats::statsMap;

datum& datum::operator=(const datum& other){
  if(this == &other){ return *this;}
  /* Copy just the counter. Lock the RHS object first */
  this->lcl.lock(); /* Ensure that you are copying the latest value "atomically" */
  // other.lcl.lock(); /* Cannot lock the other object lock. It may create coherency problems */
  this->occurances = other.occurances;
  // other.lcl.unlock(); /* Cannot lock the other object lock. It may create coherency problems */
  this->lcl.unlock();
  return *this;
}

namespace
{

  struct dummyStatsPass
      : public PassWrapper<dummyStatsPass, FunctionPass>
  {
    static char ID;
    void runOnFunction() final;
  };
} // namespace

void dummyStatsPass::runOnFunction()
{
  auto func = getFunction();
  llvm::errs() << "Dummy Stats run on function" << __FILE__ << " " << __LINE__ << " ";
  for (Block &B : func.body()){
    for (Operation &op : B){
        //auto lops = op.getOpOperands();
        //llvm::errs() << "Ops :: " << op.getName() << "Operands ::" << op.getNumOperands() << "Results :: " << op.getNumResults()  << "\n";
        std::string opname = op.getName().getStringRef().str();
        if (statsMap.find(opname) == statsMap.end()){
          datum tmp;
          statsMap[opname] = tmp;
        }
        statsMap[opname].incOccurances();

        /*if(!op.getAttrs().empty()){
            llvm::errs() << "Attributes " << op.getAttrs().size() << "\n Attributes \n";
            for(NamedAttribute attr : op.getAttrs())
                llvm::errs() << " - " << attr.first << " : " << attr.second << "\n";
            llvm::errs() << "\n";
        }*/
    }
  }
  llvm::errs() << "Dummy Stats ends on function" << __FILE__ << " " << __LINE__ << " ";
}

// create all the passes.
//

//char dummyStats::ID = 0

//static RegisterPass<dummyStats> X("dummyStats", "Dummy Statistics Pass", true, false);

std::unique_ptr<Pass> mlir::dummyStats::createDummyStatsPass()
{
  //llvm::errs() << "Dummy stats init" << __FILE__ << " " << __LINE__ << " ";
  return std::make_unique<dummyStatsPass>();
}

