#pragma once
#include "mlir/Pass/Pass.h"

#include <memory>

namespace voila::mlir
{
    namespace lowering
    {
        class PredicationForwardingPass
            : public ::mlir::PassWrapper<PredicationForwardingPass, ::mlir::OperationPass<::mlir::FuncOp>>
        {
            [[nodiscard]] ::mlir::StringRef getArgument() const final;
            [[nodiscard]] ::mlir::StringRef getDescription() const final;
            void runOnOperation() override;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createPredicationForwardingPass();
} // namespace voila::mlir
