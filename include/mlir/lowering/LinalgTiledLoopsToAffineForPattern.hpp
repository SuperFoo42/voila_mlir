#pragma once
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace voila::mlir
{
    class LinalgTiledLoopsToAffineForPattern : public ::mlir::OpRewritePattern<::mlir::linalg::TiledLoopOp>
    {
        using OpRewritePattern<::mlir::linalg::TiledLoopOp>::OpRewritePattern;

        ::mlir::LogicalResult matchAndRewrite(::mlir::linalg::TiledLoopOp tiledLoop,
                                              ::mlir::PatternRewriter &rewriter) const override;
    };

    void populateLinalgTiledLoopsToAffineForPatterns(::mlir::RewritePatternSet &patterns);

}; // namespace voila::mlir