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

    ::mlir::LogicalResult ScatterOpLowering::matchAndRewrite(ScatterOp op,
                                                             OpAdaptor adaptor,
                                                             ::mlir::ConversionPatternRewriter &rewriter) const
    {
        Value out = rewriter.create<tensor::EmptyOp>(
            op->getLoc(), op.getSrc().getType(),
            rewriter.create<tensor::DimOp>(op->getLoc(), op.getSrc(), 0)->getResults());

        rewriter.replaceOpWithNewOp<tensor::ScatterOp>(op.getOperation(), op.getSrc().getType(), op.getSrc(), out,
                                                       op.getIdxs(), rewriter.getDenseI64ArrayAttr(1),
                                                       /*TODO: can we guarantee that we have unique indices?*/ true);
        return success();
    }
} // namespace voila::mlir::lowering