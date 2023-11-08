#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

namespace comet{
#define GEN_PASS_DEF_INDEXTREEDOMAININFERENCE
#include "comet/Dialect/IndexTree/Passes.h.inc"
}

/// Define the main class as deriving from the generated base class.
struct IndexTreeDomainInference : comet::impl::IndexTreeDomainInferenceBase<IndexTreeDomainInference> {
  using IndexTreeDomainInferenceBase::IndexTreeDomainInferenceBase;

  /// The definitions of the options and statistics are now generated within
  /// the base class, but are accessible in the same way.
};