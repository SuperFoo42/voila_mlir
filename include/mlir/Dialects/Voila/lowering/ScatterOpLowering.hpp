#pragma once

#include "mlir/Support/LogicalResult.h"        // for LogicalResult
#include "mlir/Transforms/DialectConversion.h" // for ConversionPattern
#include "llvm/ADT/ArrayRef.h"                 // for ArrayRef
namespace mlir
{
    class MLIRContext;
    class Operation;
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    struct ScatterOpLowering : public ::mlir::ConversionPattern
    {
        explicit ScatterOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
