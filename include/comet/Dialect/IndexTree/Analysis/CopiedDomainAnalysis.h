#ifndef COMET_DIALECT_INDEXTREE_ANALYSIS_H
#define COMET_DIALECT_INDEXTREE_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallSet.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"

namespace mlir {
    namespace indexTree {
        struct CopiedDomainAnalysis {
            public:
            CopiedDomainAnalysis(Operation* op);
            bool isCopiedDomain(Value tensor, unsigned dim);
            bool isReductionVar(IndexTreeComputeOp op, Value index_var);

            private:
            llvm::SmallDenseSet<std::pair<Value, uint32_t>> copiedDomains;
            llvm::SmallDenseSet<std::pair<IndexTreeComputeOp, Value>> reductionVars;
            void analyzeDomains(IndexTreeComputeOp compute_op);
        };
        
    }
}

#endif // COMET_DIALECT_INDEXTREE_ANALYSIS_H