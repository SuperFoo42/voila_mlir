#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"
#include "BinaryOpLowering.hpp"

namespace voila::mlir::lowering
{
    class MulOpLowering : public ::mlir::ConversionPattern
    {
      public:
        explicit MulOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(::mlir::voila::MulOp::getOperationName(), 1, ctx) {}

        ::mlir::LogicalResult
        matchAndRewrite(::mlir::Operation *op, llvm::ArrayRef<::mlir::Value> operands, ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            auto tensorType = (*op->result_type_begin()).template dyn_cast<::mlir::TensorType>();
            ::mlir::voila::MulOp::Adaptor binaryAdaptor(operands);
            ::mlir::Value lhs, rhs;
            if (!binaryAdaptor.lhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::SmallVector<::mlir::Value> singleValueRange;
                singleValueRange.push_back(binaryAdaptor.lhs());
                lhs = rewriter.template create<::mlir::SplatOp>(loc, binaryAdaptor.lhs(), ::mlir::RankedTensorType::get(1, rewriter.getI1Type()));
            }
            else
            {
                lhs = binaryAdaptor.lhs();
            }

            if (!binaryAdaptor.rhs().getType().template isa<::mlir::TensorType>())
            {
                ::mlir::SmallVector<::mlir::Value> singleValueRange;
                singleValueRange.push_back(binaryAdaptor.rhs());
                rhs = rewriter.template create<::mlir::SplatOp>(loc, binaryAdaptor.rhs(),::mlir::RankedTensorType::get(1, rewriter.getI1Type()));
            }
            else
            {
                rhs = binaryAdaptor.rhs();
            }
            rewriter.template replaceOpWithNewOp<::mlir::tosa::MulOp>(op, tensorType, lhs, rhs, 0);
            return ::mlir::success();
        }
    };

    using AddOpLowering = BinaryOpLowering<::mlir::voila::AddOp, ::mlir::tosa::AddOp>;
    using SubOpLowering = BinaryOpLowering<::mlir::voila::SubOp, ::mlir::tosa::SubOp>;
    using DivOpLowering = BinaryOpLowering<::mlir::voila::DivOp, ::mlir::tosa::DivOp>;

} // namespace voila::mlir::lowering