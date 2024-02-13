#ifndef INDEXTREE_PATTERNS
#define INDEXTREE_PATTERNS

include "mlir/IR/PatternBase.td"
include "comet/Dialect/IndexTree/IR/IndexTreeOps.td"

def SimplifyGetPosSizePattern : Pat<
(SymbolicDomainGetPosSize(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $pos_size)>;

def SimplifyGetPosAllocPattern : Pat<
(SymbolicDomainGetPosAllocSize(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $pos_alloc_size)>;

def SimplifyGetCrdSizePattern : Pat<
(SymbolicDomainGetCrdSize(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $crd_size)>;

def SimplifyGetDimSizePattern : Pat<
(SymbolicDomainGetDimSize(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $dim_size)>;

def SimplifyGetPosPattern : Pat<
(SymbolicDomainGetPos(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $pos)>;

def SimplifyGetMarkArrayPattern : Pat<
(SymbolicDomainGetMarkArray(
    SymbolicDomainConstructOp $pos_size, $pos_alloc_size, $crd_size, $dim_size, $pos, $mark_array, $is_dynamic)),
(replaceWithValue $mark_array)>;

#endif //INDEXTREE_PATTERNS