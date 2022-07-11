#pragma once

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
//#include "mlir/lowering/LinalgTiledLoopsToAffineForPattern.hpp"


namespace mlir::func {
    class FuncOp;
}
namespace voila::mlir {
    namespace lowering {
        class LinalgTiledLoopsToAffineForPass
                : public ::mlir::PassWrapper<LinalgTiledLoopsToAffineForPass, ::mlir::OperationPass<::mlir::func::FuncOp>> {
        public:
            LinalgTiledLoopsToAffineForPass() = default;

            LinalgTiledLoopsToAffineForPass(const LinalgTiledLoopsToAffineForPass &pass) = default;

            void getDependentDialects(::mlir::DialectRegistry &registry) const final;

            [[nodiscard]] ::mlir::StringRef getDescription() const final;

            void runOnOperation() final;
        };
    }

    std::unique_ptr<::mlir::Pass> createConvertLinalgTiledLoopsToAffineForPass();
}