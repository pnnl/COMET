#include "comet/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFWHILETOFOR
#include "comet/Transforms/CometTransforms.h.inc"
}

using namespace mlir;
using namespace mlir::arith;

namespace {
    struct SCFWhileToFor : public mlir::impl::SCFWhileToForBase<SCFWhileToFor> {
        void runOnOperation() override;
    };
}

Operation *findStore(Value allocValue) {
    auto userIt = llvm::find_if(allocValue.getUsers(), [&](Operation *user) {
        auto effectInterface = dyn_cast<MemoryEffectOpInterface>(user);
        if (!effectInterface)
            return false;
        // Try to find a free effect that is applied to one of our values
        // that will be automatically freed by our pass.
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effectInterface.getEffectsOnValue(allocValue, effects);
        return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &it) {
            return isa<MemoryEffects::Write>(it.getEffect());
        });
    });
    // Assign the associated dealloc operation (if any).
    return userIt != allocValue.user_end() ? *userIt : nullptr;
}

std::tuple<unsigned int, Value> findIterArgAndRange(scf::WhileOp whileOp) {
    Region &conditionRegion = whileOp->getRegion(0);
    scf::ConditionOp conditionOp = whileOp.getConditionOp();
    Operation *conditionDefiningOp = conditionOp.getCondition().getDefiningOp();
    Operation *cmpOp = nullptr;
    if (memref::LoadOp loadOp = llvm::dyn_cast<memref::LoadOp>(conditionDefiningOp)) {
        Operation *store = findStore(loadOp.getMemRef());
        if (memref::StoreOp storeOp = llvm::dyn_cast<memref::StoreOp>(store)) {
            Operation *storedValueSource = storeOp.getOperand(0).getDefiningOp();
            if (isa<CmpIOp>(storedValueSource)) {
                cmpOp = storedValueSource;
            }
        }
    } else if (isa<CmpIOp>(conditionDefiningOp)){
        cmpOp = conditionDefiningOp;
    }
    if (cmpOp == nullptr) {
        return {0, nullptr};
    }
    for (unsigned int i = 0; i < cmpOp->getNumOperands(); i++) {
        Value cmpOperand = cmpOp->getOperand(i);
        for (unsigned int j = 0; j < conditionRegion.getNumArguments(); j++) {
            Value whileArg = conditionRegion.getArgument(j);
            if (isa<memref::LoadOp>(cmpOperand.getDefiningOp())) {
                if (whileArg == cmpOperand.getDefiningOp()->getOperand(0)) {
                    Operation *cmpLimit = cmpOp->getOperand(i xor 1).getDefiningOp();
                    if (memref::LoadOp limitLoad = dyn_cast<memref::LoadOp>(cmpLimit)) {
                        Operation *limitStore = findStore(limitLoad.getMemRef());
                        if (isa<memref::StoreOp>(limitStore))
                            return {j, limitStore->getOperand(0)};
                    } else {
                        return {j, cmpOp->getOperand(i xor 1)};
                    }
                }
            } else if (whileArg == cmpOperand) {
                    return {j, cmpOp->getOperand(i xor 1)};
                }
        }
    }
    return {0, nullptr};
}

Value checkAddOpForStep(AddIOp addIOp, unsigned int iterArgPos) {
    for (auto addOperand : addIOp->getOperands()) {
        if (ConstantIntOp constOp = llvm::dyn_cast<ConstantIntOp>(addOperand.getDefiningOp())) {
            for (auto addUser : addIOp->getUsers()) {
                if (isa<memref::StoreOp>(addUser)) {
                    Value memrefStoredInto = addUser->getOperand(1);
                    for (auto storeUser : memrefStoredInto.getUsers()) {
                        if (isa<scf::YieldOp>(storeUser)) {
                            if (storeUser->getOperand(iterArgPos) == memrefStoredInto) {
                                return constOp.getResult();
                            }
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

Value findStepAndPrepareWhile(scf::WhileOp whileOp, unsigned int iterArgPos) {
    Value bodyIterArg = whileOp->getRegion(1).getArgument(iterArgPos);
    for (auto user : bodyIterArg.getUsers()) {
        if (isa<memref::LoadOp>(user)) {
            Value loadResult = user->getResult(0);
            for (auto loadUser : loadResult.getUsers()) {
                if (isa<AddIOp>(loadUser)) {
                    Value checkedStep = checkAddOpForStep(dyn_cast<AddIOp>(loadUser), iterArgPos);
                    if (checkedStep != nullptr) {
                        if (isa<memref::GetGlobalOp>(whileOp->getOperand(iterArgPos).getDefiningOp())) {
                            /// If the whileOp lower bound is a global memref, hoist the load outside the while and use it instead
                            dyn_cast<memref::LoadOp>(user).replaceAllUsesWith(whileOp->getRegion(1).getArgument(iterArgPos));
                            user->moveBefore(whileOp);
                            user->setOperand(0, whileOp->getOperand(iterArgPos));
                            whileOp->setOperand(iterArgPos, user->getResult(0));
                        }
                    }
                    return checkedStep;
                }
            }
        } else if (isa<AddIOp>(user)) {
            return checkAddOpForStep(dyn_cast<AddIOp>(user), iterArgPos);
        }
    }
    return nullptr;
}

struct WhileOpRaising : public OpRewritePattern<scf::WhileOp> {
    using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::WhileOp whileOp, PatternRewriter &rewriter) const final {
        std::tuple<unsigned int, Value> iterArgAndRange = findIterArgAndRange(whileOp);
        if (std::get<1>(iterArgAndRange) == nullptr) {
            return failure();
        }
        unsigned int iterArgPos = std::get<0>(iterArgAndRange);
        Value step = findStepAndPrepareWhile(whileOp, iterArgPos);
        if (step == nullptr) {
            return failure();
        }
        Value iterArg = whileOp->getOperand(iterArgPos);
        whileOp->eraseOperand(iterArgPos);
        auto forOp = rewriter.create<scf::ForOp>(whileOp->getLoc(), iterArg, std::get<1>(iterArgAndRange), step, whileOp->getOperands());
        auto forIterArgs = forOp.getBody()->getArguments().drop_back(0);
        for (unsigned int i = 0; i < iterArgPos; i++)
            forIterArgs[i] = forIterArgs[i + 1];
        forIterArgs[iterArgPos] = forOp.getBody()->getArgument(0);
        rewriter.mergeBlocks(&whileOp->getRegion(1).back(), forOp.getBody(), forIterArgs);
        forOp.getBody()->getTerminator()->eraseOperand(iterArgPos);
        for (unsigned int i = 0; i < 3; i++)
            if (!forOp->getOperand(i).getType().isIndex()) {
                rewriter.setInsertionPointAfter(forOp.getOperand(i).getDefiningOp());
                auto indexCastOp = rewriter.create<IndexCastOp>(rewriter.getInsertionPoint()->getLoc(), IndexType::get(getContext()), forOp->getOperand(i));
                forOp->setOperand(i, indexCastOp.getResult());
                if (i == 0) {
                    rewriter.setInsertionPointToStart(forOp.getBody());
                    auto intCastOp = rewriter.create<IndexCastOp>(rewriter.getInsertionPoint()->getLoc(), IntegerType::get(getContext(), 32), forOp.getBody()->getArgument(0));
                    forOp.getBody()->getArgument(0).replaceAllUsesExcept(intCastOp.getResult(), llvm::SmallPtrSet<Operation *, 1>{intCastOp});
                }
            }

        unsigned int index = 0;
        for (unsigned int i = 0; i < whileOp.getNumResults(); i++)
            if (i != iterArgPos)
                whileOp->getResult(i).replaceAllUsesWith(forOp.getResult(index++));
        if (whileOp.use_empty()) {
            whileOp->remove();
        }
        return success();
    }
};

void SCFWhileToFor::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.insert<WhileOpRaising>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::createRaiseSCFWhileToForPass() {
    return std::make_unique<SCFWhileToFor>();
}
