#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::func {
    class FuncOp;
}

namespace voila::mlir {
    namespace lowering {
        class MemOpsToAffineMemOpsConversionPass
                : public ::mlir::PassWrapper<MemOpsToAffineMemOpsConversionPass, ::mlir::OperationPass<::mlir::func::FuncOp>> {
        public:
            MemOpsToAffineMemOpsConversionPass() = default;

            MemOpsToAffineMemOpsConversionPass(const MemOpsToAffineMemOpsConversionPass &pass) = default;

            void getDependentDialects(::mlir::DialectRegistry &registry) const final;

            [[nodiscard]] ::mlir::StringRef getDescription() const final;

            void runOnOperation() final;
        };
    } // namespace lowering
    std::unique_ptr<::mlir::Pass> createMemOpsToAffineMemOpsConversionPass();
} // namespace voila::mlir
