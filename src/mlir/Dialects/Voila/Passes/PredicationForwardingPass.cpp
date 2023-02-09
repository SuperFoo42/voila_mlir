#include "mlir/Dialects/Voila/Passes/PredicationForwardingPass.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/Interfaces/PredicationOpInterface.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace mlir
{
    class MLIRContext;
}

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    namespace lowering
    {
        class PredicationToBlockersForwardingPattern : public OpRewritePattern<::mlir::voila::SelectOp>
        {
            using OpRewritePattern<::mlir::voila::SelectOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(::mlir::voila::SelectOp op, PatternRewriter &rewriter) const override
            {
                SmallVector<Operation *> uses;
                SmallVector<Operation *> toPredicate;

                for (const auto &use : op->getUses())
                    uses.push_back(use.getOwner());
                while (!uses.empty())
                {
                    auto user = uses.pop_back_val();
                    // test if replacement with select operation would produce unsafe results
                    if (isa<EmitOp>(user))
                    {
                        return failure();
                    }
                    else if (isa<AvgOp>(user) || isa<SumOp>(user) || isa<CountOp>(user) || isa<MinOp>(user) ||
                             isa<MaxOp>(user) || isa<LookupOp>(user) || isa<InsertOp>(user) ||
                             isa<HashOp>(user)) // TODO: gather and scatter, but the semantic of
                                                // predication seems unclear at this point
                    {
                        toPredicate.push_back(user);
                        continue;
                    }

                    for (auto &u : user->getUses())
                        uses.push_back(u.getOwner());
                }

                for (auto *use : toPredicate)
                {
                    assert(isa<PredicationOpInterface>(use));
                    auto predOp = dyn_cast<PredicationOpInterface>(use);
                    if (predOp.predicated())
                    {
                        // conjunction of conditions
                        auto p1 = predOp.predicated();
                        rewriter.setInsertionPoint(use);
                        auto newPred = rewriter.create<arith::AndIOp>(op->getLoc(), p1, op.getPred());
                        predOp.predicate(newPred);
                    }
                    else
                    {
                        predOp.predicate(op.getPred());
                    }
                    predOp->replaceUsesOfWith(op.getResult(), op.getValues());
                }

                op->replaceAllUsesWith(ValueRange(op.getValues()));
                rewriter.eraseOp(op);

                return success();
            }
        };

        // Do we want a second version? And what should it do?
        class PredicationForwardingPattern : public OpRewritePattern<::mlir::voila::SelectOp>
        {
            using OpRewritePattern<::mlir::voila::SelectOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(::mlir::voila::SelectOp op, PatternRewriter &rewriter) const override
            {
                SmallVector<std::reference_wrapper<OpOperand>> uses;
                SmallVector<Value> toPredicate;

                auto loc = op->getLoc();

                for (auto &use : op->getUses())
                    uses.push_back(use);
                while (!uses.empty())
                {
                    auto use = uses.pop_back_val();
                    auto *user = use.get().getOwner();
                    // test if replacement with select operation would produce unsafe results
                    if (isa<InsertOp>(user) || isa<HashOp>(user) || isa<ReadOp>(user) || isa<WriteOp>(user) ||
                        isa<LookupOp>(user) || isa<EqOp>(user) || isa<GeOp>(user) || isa<GeOp>(user) ||
                        isa<GeqOp>(user) || isa<LeOp>(user) || isa<LeqOp>(user) || isa<NeqOp>(user))
                    {
                        llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.getPred());
                        user->replaceUsesOfWith(use.get().get(), op.getValues());
                        // use.drop();
                        continue;
                    }
                    else if (isa<AddOp>(user))
                    {
                        // llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.pred());
                        auto operands = user->getOperands();
                        operands[use.get().getOperandNumber()] = op.getValues();
                        rewriter.replaceOpWithNewOp<arith::AddIOp>(user, operands);
                    }
                    else if (isa<AvgOp>(user) || isa<EmitOp>(user))
                    {
                        return failure();
                    }
                    else if (isa<SumOp>(user))
                    {
                        if (!dyn_cast<SumOp>(user).getIndices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<CountOp>(user))
                    {
                        if (!dyn_cast<CountOp>(user).getIndices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<MinOp>(user))
                    {
                        if (!dyn_cast<MinOp>(user).getIndices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<MaxOp>(user))
                    {
                        if (!dyn_cast<MaxOp>(user).getIndices())
                            continue;
                        else
                            return failure();
                    }
                    for (auto &u : user->getUses())
                        uses.push_back(u);
                }

                if (!op->getUses().empty())
                {
                    Value falseSel;
                    Value tmp;
                    if (op.getValues().getType().dyn_cast<TensorType>().hasStaticShape())
                    {
                        tmp = rewriter.create<tensor::EmptyOp>(
                            loc, op.getValues().getType().dyn_cast<TensorType>().getShape(),
                            getElementTypeOrSelf(op.getValues()));
                    }
                    else
                    {
                        auto dimShape = rewriter.create<tensor::DimOp>(loc, op.getValues(), 0)->getResultTypes();
                        tmp = rewriter.create<tensor::EmptyOp>(loc, dimShape, op->getResults());
                    }

                    if (getElementTypeOrSelf(op.getValues()).isa<IntegerType>())
                    {
                        falseSel = rewriter
                                       .create<linalg::FillOp>(
                                           loc,
                                           rewriter
                                               .create<arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(),
                                                                             rewriter.getI64Type())
                                               .getResult(),
                                           ValueRange(tmp))
                                       .result();
                    }
                    else if (getElementTypeOrSelf(op.getValues()).isa<FloatType>())
                    {
                        falseSel =
                            rewriter
                                .create<linalg::FillOp>(loc,
                                                        ValueRange(rewriter
                                                                       .create<arith::ConstantFloatOp>(
                                                                           loc, rewriter.getF64FloatAttr(0).getValue(),
                                                                           rewriter.getF64Type())
                                                                       .getResult()),
                                                        ValueRange(tmp))
                                .result();
                    }
                    else
                    {
                        throw std::logic_error("Invalid type"); // TODO
                    }
                    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(op, op.getPred(), op.getValues(), falseSel);
                }

                return failure();
            }
        };

        void populatePredicationForwardingPattern(RewritePatternSet &patterns, bool predicateBlockersOnly)
        {
            auto context = patterns.getContext();
            if (predicateBlockersOnly)
                patterns.add<PredicationToBlockersForwardingPattern>(context);
            else
                patterns.add<PredicationForwardingPattern>(context);
        }

        StringRef PredicationForwardingPass::getArgument() const { return "predication-forwarding-pass"; }
        StringRef PredicationForwardingPass::getDescription() const
        {
            return "This pass tries to reduce selections with materialization by predication of operations or lazy "
                   "materialization";
        }
        void PredicationForwardingPass::runOnOperation()
        {
            RewritePatternSet patterns(&getContext());
            populatePredicationForwardingPattern(patterns, predicateBlockersOnly);
            (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
        }
    } // namespace lowering

    std::unique_ptr<Pass> createPredicationForwardingPass(bool predicateBlockersOnly)
    {
        return std::make_unique<lowering::PredicationForwardingPass>(predicateBlockersOnly);
    }
} // namespace voila::mlir

// namespace voila::mlir::lowering
