#include "mlir/Passes/PredicationForwardingPass.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    using SelectOp = ::mlir::voila::SelectOp;
    namespace lowering
    {
        class PredicationForwardingPattern : public ::mlir::OpRewritePattern<SelectOp>
        {
            using OpRewritePattern<SelectOp>::OpRewritePattern;

            ::mlir::LogicalResult matchAndRewrite(SelectOp op, PatternRewriter &rewriter) const override
            {
                SmallVector<std::reference_wrapper<OpOperand>> uses;
                SmallVector<Value> toPredicate;

                auto loc = op->getLoc();

                // TODO: transitive traverse uses and look if we want to materialize this selection or use select

                // TODO: otherwise, predicate direct uses

                // TODO: do we even need this, if fusion works correctly?

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
                        llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.pred());
                        user->replaceUsesOfWith(use.get().get(), op.values());
                        // use.drop();
                        continue;
                    }
                    else if (isa<AddOp>(user))
                    {
                        // llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.pred());
                        auto operands = user->getOperands();
                        operands[use.get().getOperandNumber()] = op.values();
                        rewriter.replaceOpWithNewOp<arith::AddIOp>(user, operands);
                    }
                    else if (isa<AvgOp>(user) || isa<EmitOp>(user))
                    {
                        return failure();
                    }
                    else if (isa<SumOp>(user))
                    {
                        if (!dyn_cast<SumOp>(user).indices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<CountOp>(user))
                    {
                        if (!dyn_cast<CountOp>(user).indices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<MinOp>(user))
                    {
                        if (!dyn_cast<MinOp>(user).indices())
                            continue;
                        else
                            return failure();
                    }
                    else if (isa<MaxOp>(user))
                    {
                        if (!dyn_cast<MaxOp>(user).indices())
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
                    if (op.values().getType().dyn_cast<TensorType>().hasStaticShape())
                    {
                        tmp = rewriter.create<linalg::InitTensorOp>(
                            loc, op.values().getType().dyn_cast<TensorType>().getShape(),
                            getElementTypeOrSelf(op.values()));
                    }
                    else
                    {
                        auto dimShape = rewriter.create<tensor::DimOp>(loc, op.values(), 0).result();
                        tmp = rewriter.create<linalg::InitTensorOp>(loc, ::llvm::makeArrayRef(dimShape),
                                                                    getElementTypeOrSelf(op.values()));
                    };

                    if (getElementTypeOrSelf(op.values()).isa<IntegerType>())
                    {
                        falseSel = rewriter
                                       .create<linalg::FillOp>(
                                           loc,
                                           rewriter.create<arith::ConstantIntOp>(
                                               loc, std::numeric_limits<int64_t>::max(), rewriter.getI64Type()),
                                           tmp)
                                       .result();
                    }
                    else if (getElementTypeOrSelf(op.values()).isa<FloatType>())
                    {
                        falseSel = rewriter
                                       .create<linalg::FillOp>(
                                           loc,
                                           rewriter.create<arith::ConstantFloatOp>(
                                               loc, rewriter.getF64FloatAttr(0).getValue(), rewriter.getF64Type()),
                                           tmp)
                                       .result();
                    }
                    else
                    {
                        throw std::logic_error("Invalid type"); // TODO
                    }
                    rewriter.replaceOpWithNewOp<::mlir::arith::SelectOp>(op, op.pred(), op.values(), falseSel);
                }

                return failure();
            }
        };

        void populatePredicationForwardingPattern(RewritePatternSet &patterns)
        {
            auto context = patterns.getContext();
            patterns.add<PredicationForwardingPattern>(context);
        }

        ::mlir::StringRef PredicationForwardingPass::getArgument() const
        {
            return "predication-forwarding-pass";
        }
        ::mlir::StringRef PredicationForwardingPass::getDescription() const
        {
            return "This pass tries to reduce selections with materialization by predication of operations or lazy "
                   "materialization";
        }
        void PredicationForwardingPass::runOnOperation()
        {
            RewritePatternSet patterns(&getContext());
            populatePredicationForwardingPattern(patterns);
        }
    } // namespace lowering
    std::unique_ptr<Pass> createPredicationForwardingPass()
    {
        return std::make_unique<lowering::PredicationForwardingPass>();
    }
} // namespace voila::mlir

// namespace voila::mlir::lowering
