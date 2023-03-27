#pragma once
#include "mlir/Dialects/Voila/IR/VoilaOps.h"   // for GatherOpAdaptor
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
    /**
     * Hash function based on XXH3 with seed = 0
     * @link{https://cyan4973.github.io/xxHash/}
     */
    struct HashOpLowering : public ::mlir::OpConversionPattern<::mlir::voila::HashOp>
    {
        using OpConversionPattern<::mlir::voila::HashOp>::OpConversionPattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::voila::HashOp op,
                                              OpAdaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering