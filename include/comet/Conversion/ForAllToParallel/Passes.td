
#ifndef FORALL_TO_PARALLEL_CONVERSION_PASSES
#define FORALL_TO_PARALLEL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ForAllToParallel: Pass<"convert-scf-forall-to-paralell", "mlir::ModuleOp"> {
    let summary = "Convert SCF forall operations to parallel";
    let description = [{

    }];
    let constructor = "mlir::comet::createForAllToParallel()";

    let dependentDialects = ["mlir::scf::SCFDialect"];
}
#endif
