#include "mlir/Dialects/Voila/lowering/GatherOpLowering.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"   // for EmptyOp, Extr...
#include "mlir/Dialects/Voila/IR/VoilaOps.h" // for GatherOpAdaptor
#include "mlir/IR/Builders.h"                // for OpBuilder
#include "mlir/IR/BuiltinTypes.h"            // for TensorType
#include "mlir/IR/Operation.h"               // for Operation
#include "mlir/IR/Types.h"                   // for Type
#include "mlir/IR/ValueRange.h"              // for ValueRange
namespace mlir
{
    class Value;
} // namespace mlir

namespace voila::mlir::lowering
{
    using namespace ::mlir;

    LogicalResult GatherOpLowering::matchAndRewrite(::mlir::voila::GatherOp op,
                                                    OpAdaptor adaptor,
                                                    ConversionPatternRewriter &rewriter) const
    {
        rewriter.replaceOpWithNewOp<tensor::GatherOp>(op.getOperation(), op.getColumn().getType(), op.getIndices(),
                                                      op.getColumn(), rewriter.getDenseI64ArrayAttr({1}),
                                                      /*TODO: can we guarantee that we have unique indices?*/ true);
        return success();
    }
} // namespace voila::mlir::lowering