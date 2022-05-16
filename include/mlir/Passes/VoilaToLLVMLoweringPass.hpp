#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

namespace mlir::func {
    class FuncOp;
}

namespace voila::mlir {
    namespace lowering {
        struct VoilaToLLVMLoweringPass
                : public ::mlir::PassWrapper<VoilaToLLVMLoweringPass, ::mlir::OperationPass<::mlir::func::FuncOp>> {
            void getDependentDialects(::mlir::DialectRegistry &registry) const override {
                registry.insert<::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect, ::mlir::async::AsyncDialect,
                        ::mlir::vector::VectorDialect, ::mlir::arith::ArithmeticDialect>();
            }

            void runOnOperation() final;
        };
    } // namespace lowering

    /**
     * Create a pass for lowering operations the remaining `Voila` operations, as
     * well as `Affine` and `Std`, to the LLVM dialect for codegen.
     * @return
     */
    std::unique_ptr<::mlir::Pass> createLowerToLLVMPass();
} // namespace voila::mlir