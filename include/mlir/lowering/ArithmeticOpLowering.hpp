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
    using AddOpLowering =
        BinaryOpLowering<::mlir::voila::AddOp, IntFloatBinOpGenerator<::mlir::arith::AddIOp, ::mlir::arith::AddFOp>>;
    using SubOpLowering =
        BinaryOpLowering<::mlir::voila::SubOp, IntFloatBinOpGenerator<::mlir::arith::SubIOp, ::mlir::arith::SubFOp>>;
    using MulOpLowering =
        BinaryOpLowering<::mlir::voila::MulOp, IntFloatBinOpGenerator<::mlir::arith::MulIOp, ::mlir::arith::MulFOp>>;
    using DivOpLowering =
        BinaryOpLowering<::mlir::voila::DivOp, IntFloatBinOpGenerator<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>>;
    using ModOpLowering =
        BinaryOpLowering<::mlir::voila::ModOp, IntFloatBinOpGenerator<::mlir::arith::RemSIOp, ::mlir::arith::RemFOp>>;

} // namespace voila::mlir::lowering