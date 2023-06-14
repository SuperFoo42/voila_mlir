#include "mlir/Dialects/Voila/lowering/LoadOpLowering.hpp"
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

    LogicalResult LoadOpLowering::matchAndRewrite(::mlir::voila::LoadOp op,
                                                    OpAdaptor adaptor,
                                                    ConversionPatternRewriter &rewriter) const
    {
        //rewriter.replaceOpWithNewOp<tensor::GatherOp>(op.getOperation(), op.getColumn().getType(), op.getIndices(),
        //                                              op.getColumn(), rewriter.getDenseI64ArrayAttr({1}),
                                                      /*TODO: can we guarantee that we have unique indices?*/ //true);
        //TODO
        return failure();
    }
} // namespace voila::mlir::lowering