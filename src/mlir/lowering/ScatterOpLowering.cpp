#include "mlir/lowering/ScatterOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::ScatterOp;
    using ::mlir::voila::ScatterOpAdaptor;
    ScatterOpLowering::ScatterOpLowering(::mlir::MLIRContext *ctx) :
        ConversionPattern(ScatterOp::getOperationName(), 1, ctx)
    {
    }

    ::mlir::LogicalResult ScatterOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                             ::mlir::ArrayRef<::mlir::Value> operands,
                                                             ::mlir::ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        ScatterOpAdaptor scatterOpAdaptor(operands);
        auto tt = scatterOpAdaptor.src().getType().dyn_cast<TensorType>();

        Value out;

        if (tt.hasStaticShape())
        {
            out = rewriter.create<memref::AllocOp>(loc, MemRefType::get(tt.getShape(), tt.getElementType()));
        }
        else
        {
            out = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(tt.getShape(), tt.getElementType()),
                llvm::makeArrayRef<Value>(rewriter.create<tensor::DimOp>(loc, scatterOpAdaptor.idxs(), 0)));
        }

        auto loopFunc = [&scatterOpAdaptor, &out](OpBuilder &builder, Location loc, ValueRange vals)
        {
            Value idx = builder.create<tensor::ExtractOp>(loc, scatterOpAdaptor.idxs(), vals);
            if (!idx.getType().isIndex())
                idx = builder.create<IndexCastOp>(loc, idx, builder.getIndexType());
            auto res = builder.create<tensor::ExtractOp>(loc, scatterOpAdaptor.src(), vals).result();
            builder.create<memref::StoreOp>(loc, res, out, idx);
        };

        llvm::SmallVector<AffineMap, 2> iter_maps(2, rewriter.getDimIdentityMap());

        buildAffineLoopNest(rewriter, loc, llvm::makeArrayRef<Value>(rewriter.create<ConstantIndexOp>(loc, 0)),
                            llvm::makeArrayRef<Value>(rewriter.create<tensor::DimOp>(loc, scatterOpAdaptor.idxs(), 0)),
                            {1}, loopFunc);

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, out);
        return success();
    }
} // namespace voila::mlir::lowering