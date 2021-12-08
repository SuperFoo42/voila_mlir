#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct LoopOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<void(::mlir::PatternRewriter &rewriter,
                                                          ::mlir::ValueRange memRefOperands,
                                                          ::mlir::ValueRange loopIvs,
                                                          ::mlir::Value iter_var)>;

        explicit LoopOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering
