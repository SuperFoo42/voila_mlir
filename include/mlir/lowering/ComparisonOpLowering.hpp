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
    template<typename CmpOp, typename LoweredBinaryOp>
    class ComparisonOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<
            ::mlir::Value(::mlir::OpBuilder &rewriter, ::mlir::ValueRange memRefOperands, ::mlir::ValueRange loopIvs)>;

      public:
        explicit ComparisonOpLowering(::mlir::MLIRContext *ctx) : ConversionPattern(CmpOp::getOperationName(), 1, ctx)
        {
        }

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();
            auto tensorType = (*op->result_type_begin()).template dyn_cast<::mlir::TensorType>();
            typename CmpOp::Adaptor binaryAdaptor(operands);
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

            ::mlir::Value cmpOp = rewriter.create<LoweredBinaryOp>(loc, tensorType,lhs, rhs);
            if constexpr (std::is_same_v<::mlir::voila::NeqOp, CmpOp> || std::is_same_v<::mlir::voila::LeOp, CmpOp> ||
                          std::is_same_v<::mlir::voila::LeqOp, CmpOp>)
            {
                rewriter.template replaceOpWithNewOp<::mlir::tosa::NegateOp>(op, cmpOp.getType(), cmpOp);
            }
            else
            {
                rewriter.replaceOp(op, cmpOp);
            }
            return ::mlir::success();
        }
    };

    using EqOpLowering = ComparisonOpLowering<::mlir::voila::EqOp, ::mlir::tosa::EqualOp>;
    using NeqOpLowering = ComparisonOpLowering<::mlir::voila::NeqOp, ::mlir::tosa::EqualOp>;
    using LeOpLowering = ComparisonOpLowering<::mlir::voila::LeOp, ::mlir::tosa::GreaterEqualOp>;
    using LeqOpLowering = ComparisonOpLowering<::mlir::voila::LeqOp, ::mlir::tosa::GreaterOp>;
    using GeOpLowering = ComparisonOpLowering<::mlir::voila::GeOp, ::mlir::tosa::GreaterOp>;
    using GeqOpLowering = ComparisonOpLowering<::mlir::voila::GeqOp, ::mlir::tosa::GreaterEqualOp>;
} // namespace voila::mlir::lowering