#include "mlir/Passes/MemOpsToAffineMemOpsConversionPass.hpp"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
                    if (isa<AffineForOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<AffineForOp>(pOp).getInductionVar();
                        ivs.push_back(iv);
                    }
                    else if (isa<AffineParallelOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<AffineParallelOp>(pOp).getIVs();
                        ivs.insert(ivs.end(), iv.begin(), iv.end());
                    }
                    pOp = pOp->getBlock()->getParentOp();
                }
                if (llvm::all_of(load.getIndices(), [&ivs](auto idx) { return llvm::find(ivs, idx) != nullptr; }))
                {
                    rewriter.replaceOpWithNewOp<AffineLoadOp>(load.getOperation(), load.getMemRef(), load.getIndices());
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
                    if (isa<AffineForOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<AffineForOp>(pOp).getInductionVar();
                        ivs.push_back(iv);
                    }
                    else if (isa<AffineParallelOp>(pOp))
                    {
                        auto iv = llvm::dyn_cast<AffineParallelOp>(pOp).getIVs();
                        ivs.insert(ivs.end(), iv.begin(), iv.end());
                    }
                    pOp = pOp->getBlock()->getParentOp();
                }
                if (llvm::all_of(store.getIndices(), [&ivs](auto idx) { return llvm::find(ivs, idx) != nullptr; }))
                {
                    rewriter.replaceOpWithNewOp<AffineStoreOp>(store.getOperation(), store.getValueToStore(),
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
            registry.insert<AffineDialect, MemRefDialect>();
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
