#include "mlir/Dialects/Voila/lowering/InsertOpLowering.hpp"
#include "NotImplementedException.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "range/v3/all.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using namespace ::voila::mlir::lowering::utils;
    using ::mlir::voila::InsertOp;

    // TODO: use atomic compare exchange
    ::mlir::LogicalResult InsertOpLowering::matchAndRewrite(::mlir::voila::InsertOp op,
                                                            OpAdaptor adaptor,
                                                            ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);
        // allocate and prefill hashtable
        auto tmp = allocHashTables(builder, op.getValues());
        SmallVector<Value> hts = tmp.first;
        auto htSize = tmp.second;

        SmallVector<Value, 1> lb;
        // TODO: add separate bucket for 0 keys to not run into problems when inserting INVALID

        SmallVector<Value> hashInvalidConsts;
        for (auto val : op.getValues())
        { // we uce all ones value of data type size
            const auto &elementType = getElementTypeOrSelf(val);
            if (elementType.isIntOrFloat())
            {
                hashInvalidConsts.push_back(builder.create<BitcastOp>(
                    elementType, builder.create<ConstantIntOp>(HASH_INVALID,
                                                               elementType.getIntOrFloatBitWidth())));
            }
            else
            {
                throw NotImplementedException();
            }
        }

        lb.push_back(builder.create<ConstantIndexOp>(0));

        for (size_t i = 0; i < hts.size(); ++i)
        {
            builder.create<linalg::FillOp>(hashInvalidConsts[i], hts[i]);
        }

        // hash values
        auto modSize = builder.create<SubIOp>(htSize, builder.create<ConstantIndexOp>(1));

        // insert values in ht
        // we can not use affine loads here, as this would lead to comprehensive bufferize complain about RaW conflicts.
        auto loopFunc = [&modSize, &hts, &hashInvalidConsts, &op](ImplicitLocOpBuilder &builder, ValueRange vals)
        {
            const auto loc = builder.getLoc();
            // load values
            auto hashIdx = builder.create<IndexCastOp>(builder.getIndexType(),
                                                       builder.create<tensor::ExtractOp>(op.getHashValues(), vals));
            Value correctedHashIdx = builder.create<AndIOp>(hashIdx, modSize);

            SmallVector<Value> toStores;
            for (auto val : op.getValues())
            {
                toStores.push_back(builder.create<tensor::ExtractOp>(val, vals));
            }
            auto loop = builder.create<::mlir::scf::WhileOp>(hashIdx.getType(), correctedHashIdx);

            // condition block
            auto beforeBlock = builder.createBlock(&loop.getBefore());
            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
            ImplicitLocOpBuilder beforeBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));

            auto comparisons =
                createKeyComparisons(beforeBuilder, hts, hashInvalidConsts, toStores, beforeBlock->getArgument(0));
            beforeBuilder.create<scf::ConditionOp>(comparisons, beforeBlock->getArgument(0));
            // body block
            auto afterBlock = builder.createBlock(&loop.getAfter());
            afterBlock->addArgument(loop->getOperand(0).getType(), loc);
            ImplicitLocOpBuilder afterBuilder(loc, OpBuilder::atBlockEnd(afterBlock));

            auto nextIdx =
                afterBuilder.create<AddIOp>(afterBlock->getArgument(0), afterBuilder.create<ConstantIndexOp>(loc, 1));
            auto nextIndexWrapOver = afterBuilder.create<AndIOp>(nextIdx, modSize);
            SmallVector<Value, 1> res;
            res.push_back(nextIndexWrapOver);
            afterBuilder.create<scf::YieldOp>(res);

            // insert
            builder.setInsertionPointAfter(loop);
            for (const auto &en : llvm::enumerate(llvm::zip(toStores, hts)))
            {
                Value toStore, ht;
                std::tie(toStore, ht) = en.value();
                builder.create<memref::StoreOp>(toStore, ht, loop->getResults());
            }
        };

        affine::buildAffineLoopNest(builder, builder.getLoc(), lb,
                            builder.create<tensor::DimOp>(op.getHashValues(), 0).getResult(), {1},
                            [&](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                            {
                                ImplicitLocOpBuilder builder(loc, nestedBuilder);
                                if (op.getPred())
                                {
                                    auto pred = builder.create<tensor::ExtractOp>(op.getPred(), vals);
                                    builder.create<scf::IfOp>(pred,
                                                              [&](OpBuilder &b, Location loc)
                                                              {
                                                                  ImplicitLocOpBuilder nb(loc, b);
                                                                  loopFunc(nb, vals);
                                                                  nb.create<scf::YieldOp>();
                                                              });
                                }
                                else
                                {
                                    loopFunc(builder, vals);
                                }
                            });

        SmallVector<Value> ret;
        for (const auto &ht : hts)
        {
            ret.push_back(builder.create<ToTensorOp>(ht));
        }
        rewriter.replaceOp(op, ret);

        return success();
    }

} // namespace voila::mlir::lowering