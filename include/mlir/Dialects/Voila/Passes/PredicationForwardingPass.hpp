#pragma once
#include <memory>                          // for unique_ptr
#include "llvm/ADT/StringRef.h"            // for operator==, StringRef
#include "mlir/Dialect/Func/IR/FuncOps.h"  // for FuncOp
#include "mlir/Pass/Pass.h"                // for OperationPass, Pass (ptr o...
#include "mlir/Support/LLVM.h"             // for StringRef

namespace voila::mlir
{
    namespace lowering
    {
        class PredicationForwardingPass
    : public ::mlir::PassWrapper<PredicationForwardingPass, ::mlir::OperationPass<::mlir::func::FuncOp>>
        {
            bool predicateBlockersOnly;

          public:
            explicit PredicationForwardingPass(bool blockersOnly) : predicateBlockersOnly(blockersOnly) {}
            [[nodiscard]] ::mlir::StringRef getArgument() const final;
            [[nodiscard]] ::mlir::StringRef getDescription() const final;

            void runOnOperation() override;
        };
    } // namespace lowering

    std::unique_ptr<::mlir::Pass> createPredicationForwardingPass(bool  predicateBlockersOnly= true);
} // namespace voila::mlir
