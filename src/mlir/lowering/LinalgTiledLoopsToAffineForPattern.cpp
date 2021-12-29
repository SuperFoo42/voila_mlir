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
        llvm::SmallVector<Operation *> users;
        llvm::SmallDenseMap<Operation *, size_t> useIVMap;
        auto generateForLoopNestAndCloneBody = [&](OpBuilder &builder, Location loc)
        {
            BlockAndValueMapping bvm;
            // bvm.map(parIVs, ivs);
            bvm.map(tiledLoop.getRegionInputArgs(), tiledLoop.inputs());
            //bvm.map(tiledLoop.getRegionOutputArgs(), tiledLoop.outputs());

            llvm::SmallSetVector<Operation *, 4> loads;
            llvm::SmallVector<Value> ivs;
            for (const auto &en : llvm::enumerate(llvm::zip(tiledLoop.getRegionOutputArgs(), tiledLoop.outputs())))
            {
                Value outArg, out, iv;
                std::tie(outArg, out) = en.value();
                auto argUsers = outArg.getUsers();
                users.append(argUsers.begin(), argUsers.end());

                auto store_count = std::count_if(users.begin(), users.end(),
                                                 [](const auto &use) -> bool
                                                 { return llvm::dyn_cast_or_null<memref::StoreOp>(use) != nullptr; });
                if (store_count >
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
                    ivs.push_back(builder.create<memref::LoadOp>(loc, out));
                }
            }
            // TODO: what about pars and seqs?
            return builder.create<AffineForOp>(
                loc, redLBs, builder.getDimIdentityMap(), redUBs, builder.getDimIdentityMap(), redSteps.front(), ivs,

                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iter_args)
                {
                    bvm.map(redIVs, llvm::makeArrayRef(iv));
                    for (auto user : users)
                    {
                        auto load = llvm::dyn_cast_or_null<memref::LoadOp>(user);
                        if (load != nullptr)
                        {

                            bvm.map(load.result(), iter_args[useIVMap.lookup(user)]);
                            loads.insert(user);
                        }
                    }
                    llvm::SmallVector<Value> results;
                    for (auto &op : tiledLoop.getBody()->without_terminator())
                    {
                        auto store = llvm::dyn_cast_or_null<memref::StoreOp>(&op);
                        if (!loads.contains(&op) && store == nullptr)
                        {
                            builder.clone(op, bvm);
                        }
                        else if (store != nullptr)
                        {
                            // TODO: we have to determine to which iter arg this store
                            // belongs, therefore we can map the memref to the iv and
                            // then have to find the right idx
                            results.push_back(bvm.lookup(store.value()));
                        }
                    }
                    builder.create<AffineYieldOp>(loc, results);
                });
        };

        auto loop = generateForLoopNestAndCloneBody(rewriter, loc);

        for (const auto &res_storage : llvm::enumerate(llvm::zip(loop.results(), tiledLoop.outputs())))
        {
            Value result, storage;
            std::tie(result, storage) = res_storage.value();
            rewriter.create<memref::StoreOp>(loc, result, storage);
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