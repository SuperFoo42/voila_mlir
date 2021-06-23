#include "mlir/lowering/ReadOpLowering.hpp"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using ::mlir::voila::ReadOp;
    using ::mlir::voila::ReadOpAdaptor;

    ReadOpLowering::ReadOpLowering(MLIRContext *ctx) : ConversionPattern(ReadOp::getOperationName(), 1, ctx) {}

    static MemRefType convertTensorToMemRef(TensorType type)
    {
        assert(type.hasRank() && "expected only ranked shapes");
        return MemRefType::get(type.getShape(), type.getElementType());
    }

    LogicalResult
    ReadOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        ReadOpAdaptor readOpAdaptor(operands);
        SmallVector<Value> ops;
        auto loc = op->getLoc();
        for (auto o : op->getOperands())
        {
            ops.push_back(o);
        }

        Value col;
        // TODO: only for tensors
        if (readOpAdaptor.column().getType().isa<TensorType>())
        {
            col = rewriter.create<memref::BufferCastOp>(
                loc, convertTensorToMemRef(readOpAdaptor.column().getType().dyn_cast<TensorType>()),
                readOpAdaptor.column());
        }
        else if (readOpAdaptor.column().getType().isa<MemRefType>())
        {
            col = readOpAdaptor.column();
        }
        else
        {
            throw MLIRLoweringError();
        }

        SmallVector<Value> sizes, offsets, strides;
        strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
        offsets.push_back(rewriter.create<IndexCastOp>(loc, readOpAdaptor.index(), rewriter.getIndexType()));
        sizes.push_back(rewriter.create<memref::DimOp>(loc, col, 0));

        rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, col, offsets, sizes, strides);
        return success();
    }
} // namespace voila::mlir::lowering