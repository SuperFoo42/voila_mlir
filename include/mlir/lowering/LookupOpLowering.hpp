#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

#include <mlir/IR/Location.h>
#include <mlir/Transforms/DialectConversion.h>

namespace voila::mlir::lowering
{
    struct LookupOpLowering : public ::mlir::ConversionPattern
    {
        explicit LookupOpLowering(::mlir::MLIRContext *ctx);

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              llvm::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final;
    };
} // namespace voila::mlir::lowering