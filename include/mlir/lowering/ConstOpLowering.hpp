#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg//IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::voila;

    template<class ConstOp>
    class ConstOpLowering : public OpRewritePattern<ConstOp>
    {
      public:
        using OpRewritePattern<ConstOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ConstOp op, PatternRewriter &rewriter) const final
        {
            auto constantValue = op.value();

            Attribute valAttr;
            if constexpr (std::is_same_v<ConstOp, IntConstOp>)
            {
                valAttr = rewriter.getI64IntegerAttr(constantValue);
            }
            else if constexpr (std::is_same_v<ConstOp, FltConstOp>)
            {
                valAttr = rewriter.getF64FloatAttr(constantValue.convertToDouble());
            }
            else if constexpr (std::is_same_v<ConstOp, BoolConstOp>)
            {
                valAttr = IntegerAttr::get(rewriter.getI1Type(), constantValue);
            }
            else
            {
                return failure();
            }

            rewriter.template replaceOpWithNewOp<ConstantOp>(op,valAttr);

            return success();
        }
    };

    using IntConstOpLowering = ConstOpLowering<IntConstOp>;
    using FltConstOpLowering = ConstOpLowering<FltConstOp>;
    using BoolConstOpLowering = ConstOpLowering<BoolConstOp>;
} // namespace voila::mlir::lowering