#include "mlir/Interfaces/VoilaInlinerInterface.hpp"

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    bool VoilaInlinerInterface::isLegalToInline(Operation *, Operation *, bool) const
    {
        return true;
    }
    bool VoilaInlinerInterface::isLegalToInline(Operation *,
                                                             Region *,
                                                             bool,
                                                             BlockAndValueMapping &) const
    {
        return true;
    }
    void VoilaInlinerInterface::handleTerminator(Operation *op,
                                                              llvm::ArrayRef<Value> valuesToRepl) const
    {
        // Only "toy.return" needs to be handled here.
        auto returnOp = cast<EmitOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
    Operation *VoilaInlinerInterface::materializeCallConversion(OpBuilder &builder,
                                                                                   Value input,
                                                                                   Type resultType,
                                                                                   Location conversionLoc) const
    {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
} // namespace voila::mlir