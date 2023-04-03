#pragma once
#include "BinaryOpLowering.hpp"

namespace mlir::voila
{
    class AndOp;
    class OrOp;
} // namespace mlir::voila

namespace voila::mlir::lowering
{

    template <> struct OpConverter<::mlir::voila::AndOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::AndIOp, ::mlir::arith::AndIOp>;
    };

    template <> struct OpConverter<::mlir::voila::OrOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::OrIOp, ::mlir::arith::OrIOp>;
    };

    using AndOpLowering = BinaryOpLowering<::mlir::voila::AndOp>;
    using OrOpLowering = BinaryOpLowering<::mlir::voila::OrOp>;

    // TODO: not op lowering
} // namespace voila::mlir::lowering