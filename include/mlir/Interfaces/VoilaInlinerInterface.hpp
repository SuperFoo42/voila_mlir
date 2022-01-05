#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>
#include "mlir/IR/VoilaOps.h"
#pragma GCC diagnostic pop


namespace voila::mlir
{
    class VoilaInlinerInterface : public ::mlir::DialectInlinerInterface
    {
        using DialectInlinerInterface::DialectInlinerInterface;

        /// This hook checks to see if the given callable operation is legal to inline
        /// into the given call. For Voila this hook can simply return true, as the Voila
        /// Call operation is always inlinable.
        bool isLegalToInline(::mlir::Operation *call, ::mlir::Operation *callable, bool wouldBeCloned) const final;

        /// This hook checks to see if the given operation is legal to inline into the
        /// given region. For Voila this hook can simply return true, as all Toy
        /// operations are inlinable.
        bool isLegalToInline(::mlir::Operation *, ::mlir::Region *, bool, ::mlir::BlockAndValueMapping &) const final;

        /// This hook is called when a terminator operation has been inlined. The only
        /// terminator that we have in the Voila dialect is the return
        /// operation(toy.return). We handle the return by replacing the values
        /// previously returned by the call operation with the operands of the
        /// return.
        void handleTerminator(::mlir::Operation *op, ::mlir::ArrayRef<::mlir::Value> valuesToRepl) const final;

        ::mlir::Operation *
        materializeCallConversion(::mlir::OpBuilder &builder, ::mlir::Value input,::mlir::Type resultType, ::mlir::Location conversionLoc) const final;
    };
} // namespace voila::mlir