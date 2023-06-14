#include "mlir/Dialects/Voila/Passes/MemOpsToAffineMemOpsConversionPass.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h" // for AffineForOp
#include "mlir/Dialect/Func/IR/FuncOps.h"     // for FuncOp
#include "mlir/Dialect/MemRef/IR/MemRef.h"    // for StoreOp
#include "mlir/IR/Block.h"                    // for Block
#include "mlir/IR/DialectRegistry.h"          // for DialectRegi...
#include "mlir/IR/Operation.h"                // for Operation
#include "mlir/IR/PatternMatch.h"             // for OpRewritePa...
#include "mlir/IR/Value.h"                    // for Value, Bloc...
#include "mlir/Support/LogicalResult.h"       // for success
#include "llvm/ADT/ArrayRef.h"                // for MutableArra...
#include "llvm/ADT/STLExtras.h"               // for all_of, find
#include "llvm/ADT/SmallVector.h"             // for SmallVector
#include "llvm/ADT/Twine.h"                   // for operator+
#include "llvm/Support/Casting.h"             // for isa, dyn_cast
#include <algorithm>                          // for max

namespace mlir
{
    class MLIRContext;
}

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace memref;
    using namespace func;
    namespace lowering
    {
        class LoadOpToAffineLoadOpPattern : public ::mlir::OpRewritePattern<LoadOp>
        {
            using OpRewritePattern<LoadOp>::OpRewritePattern;

            ::mlir::LogicalResult matchAndRewrite(LoadOp load, PatternRewriter &rewriter) const override
            {
                SmallVector<Value> ivs;
                auto pOp = load.getOperation()->getBlock()->getParentOp();
                while (!isa<FuncOp>(pOp))
                {
                    if (isa<affine::AffineForOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<affine::AffineForOp>(pOp).getInductionVar();
                        ivs.push_back(iv);
                    }
                    else if (isa<affine::AffineParallelOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<affine::AffineParallelOp>(pOp).getIVs();
                        ivs.insert(ivs.end(), iv.begin(), iv.end());
                    }
                    pOp = pOp->getBlock()->getParentOp();
                }
                if (llvm::all_of(load.getIndices(), [&ivs](auto idx) { return llvm::find(ivs, idx) != nullptr; }))
                {
                    rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(load.getOperation(), load.getMemRef(), load.getIndices());
                }
                return success();
            }
        };

        class StoreOpToAffineStoreOpPattern : public ::mlir::OpRewritePattern<StoreOp>
        {
            using OpRewritePattern<StoreOp>::OpRewritePattern;

            ::mlir::LogicalResult matchAndRewrite(StoreOp store, PatternRewriter &rewriter) const override
            {
                SmallVector<Value> ivs;
                auto pOp = store.getOperation()->getBlock()->getParentOp();
                while (!isa<FuncOp>(pOp))
                {
                    if (isa<affine::AffineForOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<affine::AffineForOp>(pOp).getInductionVar();
                        ivs.push_back(iv);
                    }
                    else if (isa<affine::AffineParallelOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<affine::AffineParallelOp>(pOp).getIVs();
                        ivs.insert(ivs.end(), iv.begin(), iv.end());
                    }
                    pOp = pOp->getBlock()->getParentOp();
                }
                if (llvm::all_of(store.getIndices(), [&ivs](auto idx) { return llvm::find(ivs, idx) != nullptr; }))
                {
                    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(store.getOperation(), store.getValueToStore(),
                                                               store.getMemRef(), store.getIndices());
                }
                return success();
            }
        };

        void populateMemOpsToAffineMemOpsPattern(RewritePatternSet &patterns)
        {
            auto context = patterns.getContext();
            patterns.add<LoadOpToAffineLoadOpPattern, StoreOpToAffineStoreOpPattern>(context);
        }

        void MemOpsToAffineMemOpsConversionPass::getDependentDialects(mlir::DialectRegistry &registry) const
        {
            registry.insert<affine::AffineDialect, MemRefDialect>();
            Pass::getDependentDialects(registry);
        }
        StringRef MemOpsToAffineMemOpsConversionPass::getDescription() const
        {
            return "Replace memory load/store ops in affine loops that use affine ivs with affine load/store ops";
        }
        void MemOpsToAffineMemOpsConversionPass::runOnOperation()
        {
            RewritePatternSet patterns(&getContext());
            populateMemOpsToAffineMemOpsPattern(patterns);
        }
    } // namespace lowering
    std::unique_ptr<Pass> createMemOpsToAffineMemOpsConversionPass()
    {
        return std::make_unique<lowering::MemOpsToAffineMemOpsConversionPass>();
    }
} // namespace voila::mlir
