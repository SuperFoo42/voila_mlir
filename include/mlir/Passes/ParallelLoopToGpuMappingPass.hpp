#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir::func {
    class FuncOp;
}
namespace voila::mlir
{
    namespace lowering
    {
        // for now more or less a copy of TestGpuGreedyParallelLoopMappingPass
        class ParallelLoopToGPUMappingPass
    : public ::mlir::PassWrapper<ParallelLoopToGPUMappingPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
            void runOnOperation() override;
        };

    } // namespace lowering
    std::unique_ptr<::mlir::Pass> createParallelLoopToGPUMappingPass();
} // namespace voila::mlir