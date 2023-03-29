#include "mlir/Dialects/Voila/lowering/LookupOpLowering.hpp"
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
#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/AffineMap.h"
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
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using namespace ::voila::mlir::lowering::utils;
    using ::mlir::utils::IteratorType;
    using ::mlir::voila::LookupOp;
    using ::mlir::voila::LookupOpAdaptor;

    ::mlir::LogicalResult LookupOpLowering::matchAndRewrite(::mlir::voila::LookupOp op,
                                                            OpAdaptor adaptor,
                                                            ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();

        auto htSizes = rewriter.create<tensor::DimOp>(loc, op.getHashtables().front(), 0);
        auto modSize = rewriter.create<SubIOp>(loc, htSizes, rewriter.create<ConstantIndexOp>(loc, 1));

        SmallVector<Value> hashInvalidConsts;
        for (auto val : op.getValues())
        { // we uce all ones value of data type size
            const auto &elementType = getElementTypeOrSelf(val);
            if (elementType.isIntOrFloat())
            {
                hashInvalidConsts.push_back(rewriter.create<BitcastOp>(
                    loc, elementType,
                    rewriter.create<ConstantIntOp>(loc, HASH_INVALID, elementType.getIntOrFloatBitWidth())));
            }
            else
            {
                throw NotImplementedException();
            }
        }
        auto lookupFunc = [&](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
        {
            ImplicitLocOpBuilder builder(loc, nestedBuilder);
            auto hashVal = vals.take_front();
            vals = vals.drop_front();
            vals = vals.drop_back();
            // probing
            Value idx =
                builder.create<AndIOp>(builder.create<IndexCastOp>(builder.getIndexType(), hashVal.front()), modSize);
            // probing with do while
            // Maybe we should keep track of the max probing count during generation and iterate only so many times
            auto loop = builder.create<scf::WhileOp>(builder.getIndexType(), idx);
            // condition

            auto beforeBlock = builder.createBlock(&loop.getBefore());
            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
            auto condBuilder = ImplicitLocOpBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));
            Value probeIdx = beforeBlock->getArgument(0);

            // lookup entries
            auto comparison = createKeyComparisons(condBuilder, op.getHashtables(), hashInvalidConsts, vals, probeIdx);
            condBuilder.create<scf::ConditionOp>(comparison, probeIdx);

            // body
            auto afterBlock = builder.createBlock(&loop.getAfter());
            afterBlock->addArgument(loop->getOperands().front().getType(), loc);
            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
            Value inc = bodyBuilder.create<AndIOp>(
                loc,
                bodyBuilder.create<AddIOp>(loc, afterBlock->getArgument(0), builder.create<ConstantIndexOp>(loc, 1)),
                modSize);
            bodyBuilder.create<scf::YieldOp>(loc, ArrayRef(inc));
            builder.setInsertionPointAfter(loop);
            // check index empty

            Value resIdx = loop->getResults().front();
            SmallVector<Value> bucketVals;
            for (auto ht : op.getHashtables())
            {
                bucketVals.push_back(builder.create<tensor::ExtractOp>(ht, resIdx));
            }
            auto notEmpty = createValueCmp(builder, bucketVals, hashInvalidConsts);

            Value res = builder.create<SelectOp>(loc, notEmpty, resIdx,
                                                 builder.create<ConstantIndexOp>(std::numeric_limits<uint64_t>::max()));
            // store result
            builder.create<linalg::YieldOp>(loc, res);
        };

        llvm::SmallVector<Value> inputs(1, op.getHashes());
        inputs.append(op.getValues().begin(), op.getValues().end());
        if (op.getPred())
        {
            Value outMemref = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(getShape(op.getHashes()), rewriter.getIndexType()),
                hasStaticShape(op.getHashes()) ? ValueRange()
                                               : rewriter.create<tensor::DimOp>(loc, op.getHashes(), 0).getResult());
            buildAffineLoopNest(
                rewriter, loc, rewriter.create<ConstantIndexOp>(loc, 0).getResult(),
                rewriter.create<tensor::DimOp>(loc, op.getHashes(), 0).getResult(), ArrayRef<int64_t>(1),
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
                {
                    ImplicitLocOpBuilder builder(loc, nestedBuilder);
                    auto pred = builder.create<tensor::ExtractOp>(op.getPred(), vals);
                    builder.create<scf::IfOp>(
                        pred,
                        [&](OpBuilder &b, Location loc)
                        {
                            ImplicitLocOpBuilder nb(loc, b);
                            auto hashVal = nb.create<tensor::ExtractOp>(op.getHashes(), vals);
                            SmallVector<Value, 6> values;
                            for (auto v : op.getValues())
                            {
                                values.push_back(nb.create<tensor::ExtractOp>(v, vals));
                            }
                            // probing
                            Value idx = nb.create<AndIOp>(nb.create<IndexCastOp>(nb.getIndexType(), hashVal), modSize);
                            // probing with do while
                            // Maybe we should keep track of the max probing count during generation and iterate only so
                            // many times
                            auto loop = nb.create<scf::WhileOp>(nb.getIndexType(), idx);
                            // condition

                            auto beforeBlock = nb.createBlock(&loop.getBefore());
                            beforeBlock->addArgument(loop->getOperands().front().getType(), loc);
                            auto condBuilder = ImplicitLocOpBuilder(loc, OpBuilder::atBlockEnd(beforeBlock));
                            Value probeIdx = beforeBlock->getArgument(0);

                            // lookup entries
                            auto comparison = createKeyComparisons(condBuilder, op.getHashtables(), hashInvalidConsts,
                                                                   values, probeIdx);
                            condBuilder.create<scf::ConditionOp>(comparison, probeIdx);

                            // body
                            auto afterBlock = nb.createBlock(&loop.getAfter());
                            afterBlock->addArgument(loop->getOperands().front().getType(), loc);
                            auto bodyBuilder = OpBuilder::atBlockEnd(afterBlock);
                            Value inc = bodyBuilder.create<AndIOp>(
                                loc,
                                bodyBuilder.create<AddIOp>(loc, afterBlock->getArgument(0),
                                                           nb.create<ConstantIndexOp>(loc, 1)),
                                modSize);
                            bodyBuilder.create<scf::YieldOp>(loc, ArrayRef(inc));
                            nb.setInsertionPointAfter(loop);
                            // check index empty

                            Value resIdx = loop->getResults().front();
                            SmallVector<Value> bucketVals;
                            for (auto ht : op.getHashtables())
                            {
                                bucketVals.push_back(nb.create<tensor::ExtractOp>(ht, resIdx));
                            }
                            auto notEmpty = createValueCmp(nb, bucketVals, hashInvalidConsts);

                            Value res =
                                nb.create<SelectOp>(loc, notEmpty, resIdx,
                                                    nb.create<ConstantIndexOp>(std::numeric_limits<uint64_t>::max()));
                            // store result
                            nb.create<AffineStoreOp>(loc, res, outMemref, vals);
                            nb.create<scf::YieldOp>();
                        });
                });
            rewriter.replaceOpWithNewOp<ToTensorOp>(op, outMemref);
        }
        else
        {
            Value outTensor = rewriter.create<tensor::EmptyOp>(
                loc, getShape(op.getHashes()), rewriter.getIndexType(),
                hasStaticShape(op.getHashes()) ? ValueRange()
                                               : rewriter.create<tensor::DimOp>(loc, op.getHashes(), 0).getResult());

            llvm::SmallVector<AffineMap> indexing_maps(/*hashes+outTensor*/ 2 + op.getValues().size(),
                                                       rewriter.getDimIdentityMap());
            auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ outTensor.getType(),
                                                               /*inputs*/ inputs,
                                                               /*outputs*/ outTensor,
                                                               /*indexing maps*/ indexing_maps,
                                                               /*iterator types*/ IteratorType::parallel, lookupFunc);
            rewriter.replaceOp(op, linalgOp->getResults());
        }

        return success();
    }

} // namespace voila::mlir::lowering
