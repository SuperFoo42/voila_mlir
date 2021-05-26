#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>
#include "VoilaOps.h"
#pragma GCC diagnostic pop


namespace voila::mlir
{
    using namespace ::mlir;

    class VoilaInlinerInterface : public DialectInlinerInterface
    {
        using DialectInlinerInterface::DialectInlinerInterface;

        /// This hook checks to see if the given callable operation is legal to inline
        /// into the given call. For Toy this hook can simply return true, as the Toy
        /// Call operation is always inlinable.
        bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final;

        /// This hook checks to see if the given operation is legal to inline into the
        /// given region. For Toy this hook can simply return true, as all Toy
        /// operations are inlinable.
        bool isLegalToInline(Operation *, Region *, bool, BlockAndValueMapping &) const final;

        /// This hook is called when a terminator operation has been inlined. The only
        /// terminator that we have in the Toy dialect is the return
        /// operation(toy.return). We handle the return by replacing the values
        /// previously returned by the call operation with the operands of the
        /// return.
        void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final;

        Operation *
        materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const final;
    };
} // namespace voila::mlir