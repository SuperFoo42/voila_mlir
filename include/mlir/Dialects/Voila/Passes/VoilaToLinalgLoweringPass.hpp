#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"  // for FuncOp
#include "mlir/Dialect/Linalg/IR/Linalg.h" // for LinalgDialect
#include "mlir/Dialect/Tensor/IR/Tensor.h" // for TensorDialect
#include "mlir/IR/DialectRegistry.h"       // for DialectRegi...
#include "mlir/Pass/Pass.h"                // for OperationPass
#include "llvm/ADT/StringRef.h"            // for operator==
#include <memory>                          // for unique_ptr

namespace mlir
{
    namespace linalg
    {
        class LinalgDialect;
    }
    namespace tensor
    {
        class TensorDialect;
    }

    namespace func
    {
        class FuncOp;
    }
} // namespace mlir
namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToLinalgLoweringPass
            : public ::mlir::PassWrapper<VoilaToLinalgLoweringPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry
                    .insert<::mlir::func::FuncDialect, ::mlir::linalg::LinalgDialect, ::mlir::tensor::TensorDialect>();
            }

            void runOnOperation() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToLinalgPass();
} // namespace voila::mlir
