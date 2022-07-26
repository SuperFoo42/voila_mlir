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
        SmallVector<Value, 3> redLBs, redUBs, redIVs;
        SmallVector<int64_t, 3> redSteps;
        for (const auto &en : llvm::enumerate(llvm::zip(tiledLoop.lowerBound(), tiledLoop.upperBound(),
                                                        tiledLoop.step(), tiledLoop.getInductionVars())))
        {
            Value lb, ub, step, iv;
            std::tie(lb, ub, step, iv) = en.value();
            if (isReductionDimension(tiledLoop, en.index()))
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
        ImplicitLocOpBuilder builder(loc, rewriter);
        llvm::SmallVector<Operation *> users;
        llvm::SmallDenseMap<Operation *, size_t> useIVMap;
        auto generateForLoopNestAndCloneBody = [&](ImplicitLocOpBuilder &builder)
        {
            BlockAndValueMapping bvm;
            // bvm.map(parIVs, ivs);
            bvm.map(tiledLoop.getRegionInputArgs(), tiledLoop.inputs());
            // bvm.map(tiledLoop.getRegionOutputArgs(), tiledLoop.outputs());

            llvm::SmallSetVector<Operation *, 4> loads;
            llvm::SmallVector<Value> ivs;
            for (const auto &en : llvm::enumerate(llvm::zip(tiledLoop.getRegionOutputArgs(), tiledLoop.outputs())))
            {
                Value outArg, out;
                std::tie(outArg, out) = en.value();
                auto argUsers = outArg.getUsers(); // unordered
                users.append(argUsers.begin(), argUsers.end());

                auto store_count = std::count_if(users.begin(), users.end(),
                                                 [](const auto &use) -> bool { return isa<memref::StoreOp>(use); });
                if (store_count !=
                    1 /*TODO: inspect, if last use of value is store, otherwise we can not make it an iter_arg*/)
                {
                    bvm.map(outArg, out);
                    // TODO: instead of failing, just use as normal seqIV
                }
                else
                {
                    for (const auto &argUser : argUsers)
                    {
                        useIVMap.insert(std::make_pair(argUser, en.index()));
                    }
                    ivs.push_back(builder.create<memref::LoadOp>(out));
                }
            }
            redLBs.insert(redLBs.end(), seqLBs.begin(), seqLBs.end());
            redUBs.insert(redUBs.end(), seqUBs.begin(), seqUBs.end());
            redSteps.insert(redSteps.end(), seqSteps.begin(), seqSteps.end());
            redIVs.insert(redIVs.end(), seqIVs.begin(), seqIVs.end());
            return builder.create<AffineForOp>(
                loc, redLBs, builder.getDimIdentityMap(), redUBs, builder.getDimIdentityMap(), redSteps.front(), ivs,

                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iter_args)
                {
                    bvm.map(redIVs, llvm::makeArrayRef(iv));
                    for (auto user : users)
                    {
                        if (isa<memref::LoadOp>(user)) // TODO: any load instruction
                        {
                            auto load = dyn_cast<memref::LoadOp>(user);
                            bvm.map(load.result(), iter_args[useIVMap.lookup(user)]);
                            loads.insert(user);
                        }
                    }
                    llvm::SmallVector<Value> results;
                    for (auto &op : tiledLoop.getBody()->without_terminator())
                    {
                        if (!loads.contains(&op) && !isa<memref::StoreOp>(&op) && !isa<memref::CopyOp>(&op))
                        {
                            builder.clone(op, bvm);
                        }
                        else if (isa<memref::StoreOp>(&op))
                        {
                            // TODO: we have to determine to which iter arg this store
                            // belongs, therefore we can map the memref to the iv and
                            // then have to find the right idx
                            results.push_back(bvm.lookup(dyn_cast<memref::StoreOp>(&op).value()));
                        }
                        else if (isa<memref::CopyOp>(&op))
                        {
                            auto copy = llvm::dyn_cast<memref::CopyOp>(&op);
                            // TODO: chek if copy is related to iter arg
                            // if copy.dest is in tiledLoop. output and copy.src is now iter arg
                            if (llvm::is_contained(tiledLoop.outputs(), copy.target()) &&
                                llvm::is_contained(tiledLoop.getRegionOutputArgs(),
                                                   copy.source()) /*&& output arg is an iter arg*/)
                            {
                                // skip, we don't need this op when we use an iter arg
                            }
                            // else if
                            // then load src value and add to iter args
                            else
                                builder.clone(op, bvm);
                        }
                    }
                    builder.create<AffineYieldOp>(loc, results);
                });
        };

        auto loop = generateForLoopNestAndCloneBody(builder);

        for (const auto &res_storage : llvm::enumerate(llvm::zip(loop.results(), tiledLoop.outputs())))
        {
            Value result, storage;
            std::tie(result, storage) = res_storage.value();
            builder.create<memref::StoreOp>(result, storage);
        }

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