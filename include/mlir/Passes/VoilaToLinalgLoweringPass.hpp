#pragma once
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir
{
    class StandardOpsDialect;
    namespace linalg
    {
        class LinalgDialect;
    }
    namespace tensor
    {
        class TensorDialect;
    }
} // namespace mlir
namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToLinalgLoweringPass : public ::mlir::PassWrapper<VoilaToLinalgLoweringPass, ::mlir::OperationPass<::mlir::FuncOp>>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry
                    .insert<::mlir::StandardOpsDialect, ::mlir::linalg::LinalgDialect, ::mlir::tensor::TensorDialect>();
            }

            void runOnOperation() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToLinalgPass();
} // namespace voila::mlir
