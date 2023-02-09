#include "mlir/Dialects/Voila/lowering/ScatterOpLowering.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"            // for buildAffine...
#include "mlir/Dialect/Arith/IR/Arith.h"                 // for IndexCastOp
#include "mlir/Dialect/Bufferization/IR/Bufferization.h" // for ToTensorOp
#include "mlir/Dialect/MemRef/IR/MemRef.h"               // for AllocOp
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // for ExtractOp
#include "mlir/Dialects/Voila/IR/VoilaOps.h"             // for ScatterOpAd...
#include "mlir/IR/AffineMap.h"                           // for AffineMap
#include "mlir/IR/BuiltinTypes.h"                        // for TensorType
#include "mlir/IR/ImplicitLocOpBuilder.h"                // for ImplicitLoc...
#include "mlir/IR/Location.h"                            // for Location
#include "mlir/IR/Operation.h"                           // for Operation
#include "mlir/IR/PatternMatch.h"                        // for PatternBenefit
#include "mlir/IR/Types.h"                               // for Type
#include "mlir/IR/Value.h"                               // for Value, Type...
#include "mlir/IR/ValueRange.h"                          // for ValueRange
#include "mlir/Support/LLVM.h"                           // for mlir
#include "llvm/ADT/STLExtras.h"                          // for ValueOfRange
#include "llvm/ADT/SmallVector.h"                        // for SmallVector
#include "llvm/ADT/StringRef.h"                          // for operator==
#include "llvm/ADT/Twine.h"                              // for operator+

namespace mlir
{
    class MLIRContext;
    class OpBuilder;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::ScatterOp;
    using ::mlir::voila::ScatterOpAdaptor;
    ScatterOpLowering::ScatterOpLowering(::mlir::MLIRContext *ctx)
        : ConversionPattern(ScatterOp::getOperationName(), 1, ctx)
    {
    }

    ::mlir::LogicalResult ScatterOpLowering::matchAndRewrite(::mlir::Operation *op,
                                                             ::mlir::ArrayRef<::mlir::Value> operands,
                                                             ::mlir::ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        ScatterOpAdaptor scatterOpAdaptor(operands);
        auto tt = scatterOpAdaptor.getSrc().getType().dyn_cast<TensorType>();

        Value out;

        if (tt.hasStaticShape())
        {
            out = builder.create<memref::AllocOp>(MemRefType::get(tt.getShape(), tt.getElementType()));
        }
        else
        {
            out = builder.create<memref::AllocOp>(
                MemRefType::get(tt.getShape(), tt.getElementType()),
                builder.create<tensor::DimOp>(scatterOpAdaptor.getIdxs(), 0).getResult());
        }

        auto loopFunc = [&scatterOpAdaptor, &out](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            Value idx = builder.create<tensor::ExtractOp>(scatterOpAdaptor.getIdxs(), vals);
            if (!idx.getType().isIndex())
                idx = builder.create<IndexCastOp>(builder.getIndexType(), idx);
            auto res = builder.create<tensor::ExtractOp>(scatterOpAdaptor.getSrc(), vals).getResult();
            builder.create<memref::StoreOp>(res, out, idx);
        };

        llvm::SmallVector<AffineMap, 2> iter_maps(2, builder.getDimIdentityMap());

        buildAffineLoopNest(rewriter, loc, builder.create<ConstantIndexOp>(0).getResult(),
                            builder.create<tensor::DimOp>(scatterOpAdaptor.getIdxs(), 0).getResult(), {1}, loopFunc);

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, out);
        return success();
    }
} // namespace voila::mlir::lowering