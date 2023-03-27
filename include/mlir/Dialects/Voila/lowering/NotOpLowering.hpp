#pragma once

#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor
#include "mlir/Support/LLVM.h"                 // for function_ref
#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef

namespace mlir
{
    class ImplicitLocOpBuilder;
    class Operation;
    class ValueRange;
    class Value;
} // namespace mlir

namespace mlir
{
    class ImplicitLocOpBuilder;
}
namespace voila::mlir::lowering
{
    struct NotOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::NotOp>
    {
        using OpConversionPattern<::mlir::voila::NotOp>::OpConversionPattern;

        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(
            ::mlir::ImplicitLocOpBuilder &rewriter, ::mlir::voila::NotOp op, ::mlir::ValueRange loopIvs)>;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::NotOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering