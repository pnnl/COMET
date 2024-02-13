#ifndef INDEXTREE_TRAITS_H_
#define INDEXTREE_TRAITS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace indexTree {
    template <typename ConcreteType>
    class UnknownDomain : public ::mlir::OpTrait::TraitBase<ConcreteType, UnknownDomain> {};

} // indexTree
} // mlir

#endif //INDEXTREE_TRAITS_H_