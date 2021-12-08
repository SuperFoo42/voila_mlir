#include "mlir/lowering/MoveOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::MoveOp;
    using ::mlir::voila::MoveOpAdaptor;

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    MoveOpLowering::MoveOpLowering(MLIRContext *ctx) : ConversionPattern(MoveOp::getOperationName(), 1, ctx) {}
    ::mlir::LogicalResult
    MoveOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        MoveOpAdaptor adaptor(operands);

        Value src, dest;
        if (adaptor.input().getType().isa<TensorType>())
        {
            src = rewriter.create<ToMemrefOp>(loc, convertTensorToMemRef(adaptor.input().getType().dyn_cast<TensorType>()),adaptor.input());
        }
        else if (adaptor.input().getType().isa<MemRefType>())
        {
            src = adaptor.input();
        }
        else
        {
            throw MLIRLoweringError();
        }

        if (adaptor.out().getType().isa<TensorType>())
        {
            dest = rewriter.create<ToMemrefOp>(loc, convertTensorToMemRef(adaptor.out().getType().dyn_cast<TensorType>()),adaptor.out());
        }
        else if (adaptor.out().getType().isa<MemRefType>())
        {
            dest = adaptor.out();
        }
        else
        {
            throw MLIRLoweringError();
        }

        rewriter.replaceOpWithNewOp<memref::CopyOp>(op, src, dest);
        return success();
    }
} // namespace voila::mlir::lowering