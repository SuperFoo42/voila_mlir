#include "mlir/lowering/CountOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::CountOp;
    using ::mlir::voila::CountOpAdaptor;

    CountOpLowering::CountOpLowering(MLIRContext *ctx) : ConversionPattern(CountOp::getOperationName(), 1, ctx) {}
    LogicalResult
    CountOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        CountOpAdaptor cntOpAdaptor(operands);

        rewriter.replaceOpWithNewOp<memref::DimOp>(op, cntOpAdaptor.input(), 0);

        return success();
    }
} // namespace voila::mlir::lowering