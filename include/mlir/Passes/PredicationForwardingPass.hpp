#pragma once
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::func {
    class FuncOp;
}

namespace voila::mlir
{
    namespace lowering
    {
        class PredicationForwardingPass
    : public ::mlir::PassWrapper<PredicationForwardingPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
            bool predicateBlockersOnly;

          public:
            explicit PredicationForwardingPass(bool blockersOnly) : predicateBlockersOnly(blockersOnly) {}
            [[nodiscard]] ::mlir::StringRef getArgument() const final;
            [[nodiscard]] ::mlir::StringRef getDescription() const final;

            void runOnOperation() override;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createPredicationForwardingPass(bool  predicateBlockersOnly= true);
} // namespace voila::mlir
