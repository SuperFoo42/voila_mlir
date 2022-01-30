#pragma once
#include "mlir/Pass/Pass.h"
namespace voila::mlir
{
    namespace lowering
    {
        // for now more or less a copy of TestGpuGreedyParallelLoopMappingPass
        class ParallelLoopToGPUMappingPass
            : public ::mlir::PassWrapper<ParallelLoopToGPUMappingPass, ::mlir::OperationPass<::mlir::FuncOp>>
        {
            [[nodiscard]] ::mlir::StringRef getArgument() const final;
            [[nodiscard]] ::mlir::StringRef getDescription() const final;
            void runOnOperation() override;
        };

    } // namespace lowering
    std::unique_ptr<::mlir::Pass> createParallelLoopToGPUMappingPass();
} // namespace voila::mlir