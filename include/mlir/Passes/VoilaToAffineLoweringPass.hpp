#pragma once
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
    class AffineDialect;
    class StandardOpsDialect;
    namespace memref {
        class MemRefDialect;
    }
    namespace bufferization{
        class BufferizationDialect;
    }
    namespace scf
    {
        class SCFDialect;
    }
    namespace linalg
    {
        class LinalgDialect;
    }
}

namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToAffineLoweringPass : public ::mlir::PassWrapper<VoilaToAffineLoweringPass, ::mlir::FunctionPass>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry.insert<::mlir::AffineDialect, ::mlir::memref::MemRefDialect, ::mlir::StandardOpsDialect,
                                ::mlir::linalg::LinalgDialect, ::mlir::scf::SCFDialect, ::mlir::bufferization::BufferizationDialect>();
            }
            void runOnFunction() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToAffinePass();
} // namespace voila::mlir