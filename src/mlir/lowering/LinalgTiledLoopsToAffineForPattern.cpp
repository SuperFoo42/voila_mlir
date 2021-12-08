#include "mlir/lowering/LinalgTiledLoopsToAffineForPattern.hpp"

namespace voila::mlir
{
    using namespace ::mlir;
    using namespace ::mlir::linalg;

    static bool isReductionDimension(TiledLoopOp &tiledLoop, const size_t dim)
    {
        return tiledLoop.iterator_types()[dim].cast<StringAttr>().getValue() == getReductionIteratorTypeName();
    }

    mlir::LogicalResult
    voila::mlir::LinalgTiledLoopsToAffineForPattern::matchAndRewrite(mlir::linalg::TiledLoopOp tiledLoop,
                                                                     mlir::PatternRewriter &rewriter) const
    {
        // Fail conversion if the `tiled_loop` has not been bufferized.
        if (!tiledLoop.hasBufferSemantics())
            return failure();

        // Collect loop control parameters for parallel, reduction and sequential dimensions.
        SmallVector<Value, 3> seqLBs, seqUBs, seqIVs;
        SmallVector<int64_t, 3> seqSteps;
        SmallVector<Value, 3> parLBs, parUBs, parIVs;
        SmallVector<int64_t, 3> parSteps;
        SmallVector<Value, 3> redLBs, redUBs, redIVs;
        SmallVector<int64_t, 3> redSteps;
        for (const auto &en : llvm::enumerate(llvm::zip(tiledLoop.lowerBound(), tiledLoop.upperBound(),
                                                        tiledLoop.step(), tiledLoop.getInductionVars())))
        {
            Value lb, ub, step, iv;
            std::tie(lb, ub, step, iv) = en.value();
            if (tiledLoop.isParallelDimension(en.index()))
            {
                parLBs.push_back(lb);
                parUBs.push_back(ub);
                auto stepConst = getConstantIntValue(getAsOpFoldResult(step));
                if (!stepConst)
                    return failure();

                parSteps.push_back(*stepConst);
                parIVs.push_back(iv);
            }
            else if (isReductionDimension(tiledLoop, en.index()))
            {
                redLBs.push_back(lb);
                redUBs.push_back(ub);
                auto stepConst = getConstantIntValue(getAsOpFoldResult(step));
                if (!stepConst)
                    return failure();

                redSteps.push_back(*stepConst);
                redIVs.push_back(iv);
            }
            else
            {
                seqLBs.push_back(lb);
                seqUBs.push_back(ub);
                auto stepConst = getConstantIntValue(getAsOpFoldResult(step));
                if (!stepConst)
                    return failure();

                seqSteps.push_back(*stepConst);
                seqIVs.push_back(iv);
            }
        }

        Location loc = tiledLoop.getLoc();
        auto generateForLoopNestAndCloneBody = [&](OpBuilder &builder, Location loc, ValueRange ivs)
        {
            BlockAndValueMapping bvm;
            //bvm.map(parIVs, ivs);
            bvm.map(tiledLoop.getRegionInputArgs(), tiledLoop.inputs());
            bvm.map(tiledLoop.getRegionOutputArgs(), tiledLoop.outputs());
//TODO: what about pars and seqs?
            buildAffineLoopNest(builder, loc, redLBs, redUBs, redSteps,
                                [&](OpBuilder &builder, Location loc, ValueRange ivs)
                                {
                                    //bvm.map(seqIVs, ivs);

                                    bvm.map(redIVs, ivs);
                                    for (auto &op : tiledLoop.getBody()->without_terminator())
                                        builder.clone(op, bvm);
                                });
            // TODO
            //  builder.setInsertionPointToStart(nest.loops.back().getBody());
        };

        generateForLoopNestAndCloneBody(rewriter, loc, llvm::None);

        rewriter.eraseOp(tiledLoop);
        return success();
    }

    void populateLinalgTiledLoopsToAffineForPatterns(RewritePatternSet &patterns)
    {
        auto context = patterns.getContext();
        AffineApplyOp::getCanonicalizationPatterns(patterns, context);
        patterns.add<LinalgTiledLoopsToAffineForPattern>(context);
    }
} // namespace voila::mlir