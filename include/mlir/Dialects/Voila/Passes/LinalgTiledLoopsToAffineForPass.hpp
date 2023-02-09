#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h" // for FuncOp
#include "mlir/Pass/Pass.h"               // for OperationPass, Pass (ptr o...
#include "mlir/Support/LLVM.h"            // for StringRef
#include "llvm/ADT/StringRef.h"           // for operator==, StringRef
#include <memory>                         // for unique_ptr

namespace mlir
{
    class DialectRegistry;
    namespace func
    {
        class FuncOp;
    }
} // namespace mlir

namespace voila::mlir
{
    namespace lowering
    {
        class LinalgTiledLoopsToAffineForPass
            : public ::mlir::PassWrapper<LinalgTiledLoopsToAffineForPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
          public:
            LinalgTiledLoopsToAffineForPass() = default;

            LinalgTiledLoopsToAffineForPass(const LinalgTiledLoopsToAffineForPass &pass) = default;

            void getDependentDialects(::mlir::DialectRegistry &registry) const final;

            [[nodiscard]] ::mlir::StringRef getDescription() const final;

            void runOnOperation() final;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createConvertLinalgTiledLoopsToAffineForPass();
} // namespace voila::mlir