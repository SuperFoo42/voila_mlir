#pragma once
#include "BinaryOpLowering.hpp"

namespace mlir::voila
{
    class AndOp;
    class OrOp;
} // namespace mlir::voila

namespace voila::mlir::lowering
{
    using AndOpLowering = BinaryOpLowering<::mlir::voila::AndOp, SingleTypeBinOpGenerator<::mlir::arith::AndIOp>>;
    using OrOpLowering = BinaryOpLowering<::mlir::voila::OrOp, SingleTypeBinOpGenerator<::mlir::arith::OrIOp>>;

    // TODO: not op lowering
} // namespace voila::mlir::lowering