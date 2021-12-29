#pragma once
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "llvm/ADT/Sequence.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct SumOpLowering : public ::mlir::ConversionPattern
    {
        using LoopIterationFn = ::mlir::function_ref<::mlir::Value(::mlir::OpBuilder &rewriter,
                                                                   ::mlir::ValueRange memRefOperands,
                                                                   ::mlir::ValueRange loopIvs,
                                                                   ::mlir::Value iter_var)>;
        explicit SumOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering