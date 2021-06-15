#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"


#include "llvm/ADT/Sequence.h"
namespace voila::mlir::lowering
{
    template<typename BinaryOp, typename LoweredBinaryOp>
    class BinaryOpLowering : public ::mlir::ConversionPattern
    {
      public:
        explicit BinaryOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

        ::mlir::LogicalResult
        matchAndRewrite(::mlir::Operation *op, llvm::ArrayRef<::mlir::Value> operands, ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            auto tensorType = (*op->result_type_begin()).template dyn_cast<::mlir::TensorType>();
            typename BinaryOp::Adaptor binaryAdaptor(operands);
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
            rewriter.template replaceOpWithNewOp<LoweredBinaryOp>(op, tensorType, lhs, rhs);
            return ::mlir::success();
        }
    };
} // namespace voila::mlir::lowering