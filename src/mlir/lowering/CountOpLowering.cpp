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
        auto loc = op->getLoc();
        CountOpAdaptor cntOpAdaptor(operands);

        auto cnt = rewriter.create<tensor::DimOp>(loc, cntOpAdaptor.input(), 0);
        rewriter.replaceOpWithNewOp<IndexCastOp>(op, cnt, rewriter.getI64Type());

        return success();
    }
} // namespace voila::mlir::lowering