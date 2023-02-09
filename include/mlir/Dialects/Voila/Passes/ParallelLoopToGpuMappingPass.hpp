#pragma once
#include <memory>                          // for unique_ptr
#include "llvm/ADT/StringRef.h"            // for operator==
#include "mlir/Dialect/Func/IR/FuncOps.h"  // for FuncOp
#include "mlir/Pass/Pass.h"                // for OperationPass, Pass (ptr o...

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