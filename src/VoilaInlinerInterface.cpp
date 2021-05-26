#include "mlir/VoilaInlinerInterface.hpp"
bool voila::mlir::VoilaInlinerInterface::isLegalToInline(mlir::Operation *, mlir::Operation *, bool) const
{
    return true;
}
bool voila::mlir::VoilaInlinerInterface::isLegalToInline(mlir::Operation *,
                                                         mlir::Region *,
                                                         bool,
                                                         mlir::BlockAndValueMapping &) const
{
    return true;
}
void voila::mlir::VoilaInlinerInterface::handleTerminator(mlir::Operation *op,
                                                          llvm::ArrayRef<mlir::Value> valuesToRepl) const
{
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<::mlir::voila::EmitOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
        valuesToRepl[it.index()].replaceAllUsesWith(it.value());
}
mlir::Operation *voila::mlir::VoilaInlinerInterface::materializeCallConversion(mlir::OpBuilder &builder,
                                                                               mlir::Value input,
                                                                               mlir::Type resultType,
                                                                               mlir::Location conversionLoc) const
{
    return builder.create<mlir::voila::CastOp>(conversionLoc, resultType, input);
}
