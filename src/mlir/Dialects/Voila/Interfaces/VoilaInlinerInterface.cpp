#include "mlir/Dialects/Voila/Interfaces/VoilaInlinerInterface.hpp"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/Interfaces/PredicationOpInterface.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <cassert>

namespace mlir
{
    class IRMapping;
    class Region;
} // namespace mlir

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::voila;
    bool VoilaInlinerInterface::isLegalToInline(Operation *, Operation *, bool) const { return true; }
    bool VoilaInlinerInterface::isLegalToInline(Operation *, Region *, bool, IRMapping &) const { return true; }
    void VoilaInlinerInterface::handleTerminator(Operation *op, ValueRange valuesToRepl) const
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