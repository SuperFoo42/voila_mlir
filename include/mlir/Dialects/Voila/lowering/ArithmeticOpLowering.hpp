#pragma once
#include "BinaryOpLowering.hpp"

namespace mlir::voila
{
    class AddOp;
    class SubOp;
    class DivOp;
    class MulOp;
    class ModOp;
} // namespace mlir::voila

namespace voila::mlir::lowering
{
    template <> struct OpConverter<::mlir::voila::AddOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::AddIOp, ::mlir::arith::AddFOp>;
    };

    template <> struct OpConverter<::mlir::voila::SubOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::SubIOp, ::mlir::arith::SubFOp>;
    };

    template <> struct OpConverter<::mlir::voila::MulOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::MulIOp, ::mlir::arith::MulFOp>;
    };
    ;

    template <> struct OpConverter<::mlir::voila::DivOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>;
    };

    template <> struct OpConverter<::mlir::voila::ModOp>
    {
        constexpr static auto createOp = createBinOp<::mlir::arith::RemSIOp, ::mlir::arith::RemFOp>;
    };

    using AddOpLowering = BinaryOpLowering<::mlir::voila::AddOp>;
    using SubOpLowering = BinaryOpLowering<::mlir::voila::SubOp>;
    using MulOpLowering = BinaryOpLowering<::mlir::voila::MulOp>;
    using DivOpLowering = BinaryOpLowering<::mlir::voila::DivOp>;
    using ModOpLowering = BinaryOpLowering<::mlir::voila::ModOp>;

} // namespace voila::mlir::lowering