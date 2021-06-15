#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/VoilaDialect.h"

#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace voila::mlir::lowering
{
    struct VoilaToAffineLoweringPass : public ::mlir::PassWrapper<VoilaToAffineLoweringPass, ::mlir::FunctionPass>
    {
        void getDependentDialects(::mlir::DialectRegistry &registry) const override
        {
            registry.insert<::mlir::AffineDialect, ::mlir::memref::MemRefDialect, ::mlir::StandardOpsDialect, ::mlir::tosa::TosaDialect, ::mlir::scf::SCFDialect, ::mlir::tensor::TensorDialect>();
        }
        void runOnFunction() final;
    };
} // namespace voila::mlir::lowering

namespace voila::mlir
{
    std::unique_ptr<::mlir::Pass> createLowerToAffinePass();
}