#include "mlir/Dialects/Voila/lowering/utility/HashingUtils.hpp"
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

namespace voila::mlir::lowering::utils
{

    using namespace ::mlir;
    using namespace arith;
    using namespace bufferization;
    using ::mlir::voila::InsertOp;

    Value createValueCmp(ImplicitLocOpBuilder &builder, const SmallVector<Value> &vals, const SmallVector<Value> &toCmp)
    {
        auto cmpView = ranges::transform2_view(
            vals, toCmp,
            [&builder](auto bv, auto inv) -> Value
            {
                if (inv.getType().template isa<IntegerType>())
                {
                    return builder.create<CmpIOp>(builder.getI1Type(), CmpIPredicate::ne, bv, inv);
                }
                else if (inv.getType().template isa<FloatType>())
                {
                    return builder.create<CmpFOp>(builder.getI1Type(), CmpFPredicate::ONE, bv, inv);
                }
                else
                {
                    throw NotImplementedException();
                }
            });

        auto fstCmp = ranges::take_view(cmpView, 1);

        auto anyCmp = ranges::accumulate(ranges::drop_view(cmpView, 1), *fstCmp.begin(),
                                         [&builder](auto l, auto r) -> Value { return builder.create<OrIOp>(l, r); });
        return anyCmp;
    }

    std::pair<SmallVector<Value>, Value> allocHashTables(ImplicitLocOpBuilder &rewriter, ValueRange values)
    {
        Value htSize;
        SmallVector<Value> hts;
        auto valType = values.front().getType().dyn_cast<TensorType>();
        // can calculate ht size from static shape
        if (valType.hasStaticShape())
        {
            auto size = std::bit_ceil<size_t>(valType.getShape().front() + 1);
            assert(size <= std::numeric_limits<int64_t>::max());
            htSize = rewriter.create<ConstantIndexOp>(size);
        }
        else
        {
            auto insertSize = rewriter.create<tensor::DimOp>(values.front(), 0);
            /** algorithm to find the next power of 2 taken from
             *  https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
             *
             * v |= v >> 1;
             * v |= v >> 2;
             * v |= v >> 4;
             * v |= v >> 8;
             * v |= v >> 16;
             * v |= v >> 32;
             * v++;
             */
            auto firstOr = rewriter.create<OrIOp>(
                insertSize, rewriter.create<ShRUIOp>(insertSize, rewriter.create<ConstantIndexOp>(1)));
            auto secondOr =
                rewriter.create<OrIOp>(firstOr, rewriter.create<ShRUIOp>(firstOr, rewriter.create<ConstantIndexOp>(2)));
            auto thirdOr = rewriter.create<OrIOp>(
                secondOr, rewriter.create<ShRUIOp>(secondOr, rewriter.create<ConstantIndexOp>(4)));
            auto fourthOr =
                rewriter.create<OrIOp>(thirdOr, rewriter.create<ShRUIOp>(thirdOr, rewriter.create<ConstantIndexOp>(8)));
            auto fithOr = rewriter.create<OrIOp>(
                fourthOr, rewriter.create<ShRUIOp>(fourthOr, rewriter.create<ConstantIndexOp>(16)));
            auto sixthOr =
                rewriter.create<OrIOp>(fithOr, rewriter.create<ShRUIOp>(fithOr, rewriter.create<ConstantIndexOp>(32)));

            htSize = rewriter.create<AddIOp>(sixthOr, rewriter.create<ConstantIndexOp>(1));
        }
        for (auto val : values)
        {
            hts.push_back(rewriter.create<memref::AllocOp>(
                MemRefType::get(ShapedType::kDynamic, val.getType().dyn_cast<TensorType>().getElementType()), htSize));
        }

        return {hts, htSize};
    }

    Value createKeyComparisons(ImplicitLocOpBuilder &builder,
                              const SmallVector<Value> &hts,
                              const SmallVector<Value> &hashInvalidConsts,
                              const SmallVector<Value> &toStores,
                              const ValueRange &idx)
    {
        SmallVector<Value> bucketVals;
        ranges::transform(hts, std::back_inserter(bucketVals),
                          [&builder, &idx](const auto ht) { return builder.create<memref::LoadOp>(ht, idx); });

        auto anyNotEmpty = createValueCmp(builder, bucketVals, hashInvalidConsts);

        auto anyNotFound = createValueCmp(builder, bucketVals, toStores);
        return builder.create<AndIOp>(anyNotEmpty, anyNotFound);
    }

    Value getHTSize(ImplicitLocOpBuilder &builder, Value values)
    {
        auto insertSize = builder.create<tensor::DimOp>(values, 0);
        /** algorithm to find the next power of 2 taken from
         *  https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
         *
         * v |= v >> 1;
         * v |= v >> 2;
         * v |= v >> 4;
         * v |= v >> 8;
         * v |= v >> 16;
         * v |= v >> 32;
         * v++;
         */
        auto firstOr =
            builder.create<OrIOp>(insertSize, builder.create<ShRUIOp>(insertSize, builder.create<ConstantIndexOp>(1)));
        auto secondOr =
            builder.create<OrIOp>(firstOr, builder.create<ShRUIOp>(firstOr, builder.create<ConstantIndexOp>(2)));
        auto thirdOr =
            builder.create<OrIOp>(secondOr, builder.create<ShRUIOp>(secondOr, builder.create<ConstantIndexOp>(4)));
        auto fourthOr =
            builder.create<OrIOp>(thirdOr, builder.create<ShRUIOp>(thirdOr, builder.create<ConstantIndexOp>(8)));
        auto fithOr =
            builder.create<OrIOp>(fourthOr, builder.create<ShRUIOp>(fourthOr, builder.create<ConstantIndexOp>(16)));
        auto sixthOr =
            builder.create<OrIOp>(fithOr, builder.create<ShRUIOp>(fithOr, builder.create<ConstantIndexOp>(32)));

        return builder.create<AddIOp>(sixthOr, builder.create<ConstantIndexOp>(1));
    }
} // namespace voila::mlir::lowering::utils