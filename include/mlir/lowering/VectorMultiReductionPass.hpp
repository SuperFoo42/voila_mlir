#pragma once
#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace voila::mlir
{
    /**
     * This class is basically a copy of the TestVectorTransforms pass for vector multi-reductions
     */

    class VectorMultiReductionPass : public ::mlir::PassWrapper<VectorMultiReductionPass, ::mlir::FunctionPass>
    {
      public:
        VectorMultiReductionPass() = default;
        VectorMultiReductionPass(const VectorMultiReductionPass &pass) {}
        void getDependentDialects(::mlir::DialectRegistry &registry) const override;
        ::mlir::StringRef getArgument() const final
        {
            return "test-vector-multi-reduction-lowering-patterns";
        }
        ::mlir::StringRef getDescription() const final;
        Option<bool> useOuterReductions{*this, "use-outer-reductions",
                                        llvm::cl::desc("Move reductions to outer most dimensions"),
                                        llvm::cl::init(false)};
        void runOnFunction() override;
    };

    std::unique_ptr<::mlir::Pass> createLowerVectorMultiReductionPass();
}