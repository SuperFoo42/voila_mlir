#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace voila::mlir
{
    namespace lowering
    {
        struct VoilaToAffineLoweringPass
            : public ::mlir::PassWrapper<VoilaToAffineLoweringPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override
            {
                registry.insert<::mlir::affine::AffineDialect, ::mlir::memref::MemRefDialect, ::mlir::func::FuncDialect,
                                ::mlir::linalg::LinalgDialect, ::mlir::scf::SCFDialect,
                                ::mlir::bufferization::BufferizationDialect>();
            }

            void runOnOperation() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createLowerToAffinePass();
} // namespace voila::mlir