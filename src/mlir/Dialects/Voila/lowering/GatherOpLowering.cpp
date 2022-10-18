#include "mlir/Dialects/Voila/lowering/GatherOpLowering.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using ::mlir::voila::GatherOp;
    using ::mlir::voila::GatherOpAdaptor;

    GatherOpLowering::GatherOpLowering(MLIRContext *ctx) : ConversionPattern(GatherOp::getOperationName(), 1, ctx) {}

    LogicalResult GatherOpLowering::matchAndRewrite(Operation *op,
                                                    ArrayRef<Value> operands,
                                                    ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();

        GatherOpAdaptor gatherOpAdaptor(operands);

        auto out = rewriter
                       .create<tensor::EmptyOp>(loc, gatherOpAdaptor.getIndices().getType().dyn_cast<TensorType>().getShape(),
                                                     getElementTypeOrSelf(gatherOpAdaptor.getColumn()))
                       ->getResults();

        auto gatherFunc = [&gatherOpAdaptor](OpBuilder & builder, Location loc, ValueRange vals)
        {
            auto idx = vals.front();
            auto res = builder.create<tensor::ExtractOp>(loc, gatherOpAdaptor.getColumn(), idx).getResult();
            builder.create<linalg::YieldOp>(loc, res);
        };

        llvm::SmallVector<AffineMap, 2> iter_maps(2, rewriter.getDimIdentityMap());

        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc, /*results*/ out.getType(),
            /*inputs*/ llvm::makeArrayRef(gatherOpAdaptor.getIndices()), /*outputs*/ out,
            /*indexing maps*/ iter_maps,
            /*iterator types*/ getParallelIteratorTypeName(), gatherFunc);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }
} // namespace voila::mlir::lowering