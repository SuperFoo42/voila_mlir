#include "mlir/Dialects/Voila/lowering/ScatterOpLowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"

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
        ImplicitLocOpBuilder builder(loc, rewriter);
        ScatterOpAdaptor scatterOpAdaptor(operands);
        auto tt = scatterOpAdaptor.src().getType().dyn_cast<TensorType>();

        Value out;

        if (tt.hasStaticShape())
        {
            out = builder.create<memref::AllocOp>(MemRefType::get(tt.getShape(), tt.getElementType()));
        }
        else
        {
            out = builder.create<memref::AllocOp>(
                MemRefType::get(tt.getShape(), tt.getElementType()),
                llvm::makeArrayRef<Value>(builder.create<tensor::DimOp>(scatterOpAdaptor.idxs(), 0)));
        }

        auto loopFunc = [&scatterOpAdaptor, &out](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            Value idx = builder.create<tensor::ExtractOp>(scatterOpAdaptor.idxs(), vals);
            if (!idx.getType().isIndex())
                idx = builder.create<IndexCastOp>(builder.getIndexType(), idx);
            auto res = builder.create<tensor::ExtractOp>(scatterOpAdaptor.src(), vals).getResult();
            builder.create<memref::StoreOp>(res, out, idx);
        };

        llvm::SmallVector<AffineMap, 2> iter_maps(2, builder.getDimIdentityMap());

        buildAffineLoopNest(rewriter, loc, llvm::makeArrayRef<Value>(builder.create<ConstantIndexOp>(0)),
                            llvm::makeArrayRef<Value>(builder.create<tensor::DimOp>(scatterOpAdaptor.idxs(), 0)), {1},
                            loopFunc);

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, out);
        return success();
    }
} // namespace voila::mlir::lowering