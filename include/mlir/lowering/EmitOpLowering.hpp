#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

namespace voila::mlir::lowering
{
    struct EmitOpLowering : public OpRewritePattern<::mlir::voila::EmitOp>
    {
        using OpRewritePattern<::mlir::voila::EmitOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(::mlir::voila::EmitOp op, PatternRewriter &rewriter) const final
        {
            // lower to std::return
            // here we should only have to deal with the emit of the main function, since all other uses should have been inlined
            rewriter.replaceOpWithNewOp<::mlir::ReturnOp>(op, op.getOperands());

            return success();
        }
    };
} // namespace voila::mlir::lowering