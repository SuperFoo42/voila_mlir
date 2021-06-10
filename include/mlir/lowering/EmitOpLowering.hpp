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
    struct EmitOpLowering : public ::mlir::OpRewritePattern<::mlir::voila::EmitOp>
    {
        ::mlir::FuncOp &function;

        EmitOpLowering(::mlir::MLIRContext *ctx, ::mlir::FuncOp &function) :
            ::mlir::OpRewritePattern<::mlir::voila::EmitOp>(ctx), function{function} {}
        //using OpRewritePattern<::mlir::voila::EmitOp>::OpRewritePattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::EmitOp op, ::mlir::PatternRewriter &rewriter) const final
        {
            // lower to std::return
            // here we should only have to deal with the emit of the main function, since all other uses should have been inlined
            ::mlir::SmallVector<::mlir::Value> ops;
            for (auto o : op.getOperands())
            {
                    ops.push_back(o);
            }

            rewriter.replaceOpWithNewOp<::mlir::ReturnOp>(op, ops);
            return ::mlir::success();
        }
    };
} // namespace voila::mlir::lowering