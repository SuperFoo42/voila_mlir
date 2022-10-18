#include "mlir/Dialects/Voila/lowering/HashOpLowering.hpp"

#include "NotImplementedException.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"

#include "fmt/format.h"
namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace arith;
    using ::mlir::voila::HashOp;
    using ::mlir::voila::HashOpAdaptor;

    HashOpLowering::HashOpLowering(MLIRContext *ctx) : ConversionPattern(HashOp::getOperationName(), 1, ctx) {}

    static constexpr auto XXH_PRIME64_1 = 0x9E3779B185EBCA87ULL;
    static constexpr auto XXH_SECRET = std::to_array<uint8_t>(
        {0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4,
         0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f, 0xcb, 0x79, 0xe6, 0x4e,
         0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43,
         0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c, 0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb,
         0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3, 0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19,
         0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8, 0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7,
         0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78,
         0x73, 0x64, 0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
         0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e, 0x2b, 0x16,
         0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f,
         0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e});

    // TODO: check if this becomes a bswap instruction when compiled
    static Value XXH_swap64(ImplicitLocOpBuilder &builder, Value x)
    {
        assert(x.getType().getIntOrFloatBitWidth() == 64);

        return builder.create<OrIOp>(
            builder.create<OrIOp>(
                builder.create<OrIOp>(
                    builder.create<OrIOp>(
                        builder.create<OrIOp>(
                            builder.create<OrIOp>(
                                builder.create<OrIOp>(
                                    builder.create<AndIOp>(
                                        builder.create<ShLIOp>(x,
                                                               builder.create<ConstantIntOp>(56, builder.getI64Type())),
                                        builder.create<ConstantIntOp>(0xff00000000000000ULL, builder.getI64Type())),
                                    builder.create<AndIOp>(
                                        builder.create<ShLIOp>(x,
                                                               builder.create<ConstantIntOp>(40, builder.getI64Type())),
                                        builder.create<ConstantIntOp>(0x00ff000000000000ULL, builder.getI64Type()))),
                                builder.create<AndIOp>(
                                    builder.create<ShLIOp>(x, builder.create<ConstantIntOp>(24, builder.getI64Type())),
                                    builder.create<ConstantIntOp>(0x0000ff0000000000ULL, builder.getI64Type()))),
                            builder.create<AndIOp>(
                                builder.create<ShLIOp>(x, builder.create<ConstantIntOp>(8, builder.getI64Type())),
                                builder.create<ConstantIntOp>(0x000000ff00000000ULL, builder.getI64Type()))),
                        builder.create<AndIOp>(
                            builder.create<ShRUIOp>(x, builder.create<ConstantIntOp>(8, builder.getI64Type())),
                            builder.create<ConstantIntOp>(0x00000000ff000000ULL, builder.getI64Type()))),
                    builder.create<AndIOp>(
                        builder.create<ShRUIOp>(x, builder.create<ConstantIntOp>(24, builder.getI64Type())),
                        builder.create<ConstantIntOp>(0x0000000000ff0000ULL, builder.getI64Type()))),
                builder.create<AndIOp>(
                    builder.create<ShRUIOp>(x, builder.create<ConstantIntOp>(40, builder.getI64Type())),
                    builder.create<ConstantIntOp>(0x000000000000ff00ULL, builder.getI64Type()))),
            builder.create<AndIOp>(builder.create<ShRUIOp>(x, builder.create<ConstantIntOp>(56, builder.getI64Type())),
                                   builder.create<ConstantIntOp>(0x00000000000000ffULL, builder.getI64Type())));
    }

    template<int WIDTH>
    static Value getINT(ImplicitLocOpBuilder &builder, Value val)
    {
        if (val.getType().getIntOrFloatBitWidth() < WIDTH)
        {
            return builder.create<ExtSIOp>(builder.getIntegerType(WIDTH), val);
        }
        else if (val.getType().getIntOrFloatBitWidth() == WIDTH)
        {
            return val;
        }
        else
        {
            throw std::logic_error(
                fmt::format("Can not hash value with {} bits", val.getType().getIntOrFloatBitWidth()));
        }
    }

    static auto split(ImplicitLocOpBuilder &builder, Value val)
    {
        if (val.getType().isa<FloatType>())
            val = builder.create<arith::BitcastOp>(builder.getIntegerType(val.getType().getIntOrFloatBitWidth()), val);

        auto lower = builder.create<arith::TruncIOp>(builder.getI32Type(), val);
        auto upper = builder.create<arith::TruncIOp>(
            builder.getI32Type(),
            builder.create<ShRUIOp>(val, builder.create<ConstantIntOp>(32, builder.getI64Type())));
        return std::make_pair(lower, upper);
    }

    static auto combine(ImplicitLocOpBuilder &builder, Value val1, Value val2)
    {
        if (val1.getType().isa<FloatType>())
        {
            val1 =
                builder.create<arith::BitcastOp>(builder.getIntegerType(val1.getType().getIntOrFloatBitWidth()), val1);
        }
        if (val2.getType().isa<FloatType>())
        {
            val2 =
                builder.create<arith::BitcastOp>(builder.getIntegerType(val2.getType().getIntOrFloatBitWidth()), val2);
        }
        auto extended = builder.create<arith::ExtUIOp>(builder.getI64Type(), val1);
        auto shifted = builder.create<arith::ShLIOp>(extended, builder.create<ConstantIntOp>(32, builder.getI64Type()));
        auto extended2 = builder.create<arith::ExtUIOp>(builder.getI64Type(), val2);
        auto combined = builder.create<arith::OrIOp>(shifted, extended2);
        return combined;
    }

    static Value XXH3_mul128_fold64(ImplicitLocOpBuilder &builder, Value lhs, Value rhs)
    {
        auto lhs128 = builder.create<ExtSIOp>(builder.getIntegerType(128), lhs);
        auto rhs128 = builder.create<ExtSIOp>(builder.getIntegerType(128), rhs);
        auto product = builder.create<MulIOp>(lhs128, rhs128);
        auto product_low = builder.create<TruncIOp>(builder.getI64Type(), product);
        auto product_high = builder.create<TruncIOp>(
            builder.getI64Type(),
            builder.create<ShRUIOp>(product, builder.create<ConstantIntOp>(64, product.getType())));

        return builder.create<XOrIOp>(product_low, product_high);
    }

    static Value XXH3_avalanche(ImplicitLocOpBuilder &builder, Value h64)
    {
        auto h64_1 = builder.create<XOrIOp>(
            h64, builder.create<ShRUIOp>(h64, builder.create<ConstantIntOp>(37, builder.getI64Type())));
        auto h64_2 =
            builder.create<MulIOp>(h64_1, builder.create<ConstantIntOp>(0x165667919E3779F9ULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            h64_2, builder.create<ShRUIOp>(h64_2, builder.create<ConstantIntOp>(32, builder.getI64Type())));
        return h64_3;
    }

    static Value XXH3_mix16B(ImplicitLocOpBuilder &builder, ValueRange vs, const size_t secret_offset)
    {
        auto lo = vs[0];
        auto hi = vs[1];
        return XXH3_mul128_fold64(
            builder,
            builder.create<XOrIOp>(
                lo, builder.create<ConstantIntOp>(*reinterpret_cast<const int64_t *>(&XXH_SECRET[secret_offset]), 64)),
            builder.create<XOrIOp>(
                hi, builder.create<ConstantIntOp>(
                        *reinterpret_cast<const int64_t *>(&XXH_SECRET[secret_offset + sizeof(int64_t)]), 64)));
    }

    static Value XXH3_rrmxmx(ImplicitLocOpBuilder &builder, Value h64, unsigned int len)
    {
        // rotate h64
        // TODO: check that this become a rotl instruction
        auto h64_r1 = builder.create<OrIOp>(
            builder.create<ShLIOp>(h64, builder.create<ConstantIntOp>(49, builder.getI64Type())),
            builder.create<ShRUIOp>(h64, builder.create<ConstantIntOp>(15, builder.getI64Type())));
        auto h64_r2 = builder.create<OrIOp>(
            builder.create<ShLIOp>(h64, builder.create<ConstantIntOp>(24, builder.getI64Type())),
            builder.create<ShRUIOp>(h64, builder.create<ConstantIntOp>(40, builder.getI64Type())));
        auto h64_1 = builder.create<XOrIOp>(h64, builder.create<XOrIOp>(h64_r1, h64_r2));
        auto h64_2 =
            builder.create<MulIOp>(h64_1, builder.create<ConstantIntOp>(0x9FB21C651E98DF25ULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            h64_2, builder.create<AddIOp>(
                       builder.create<ShRUIOp>(h64_2, builder.create<ConstantIntOp>(35, builder.getI64Type())),
                       builder.create<ConstantIntOp>(len, builder.getI64Type())));
        auto h64_4 =
            builder.create<MulIOp>(h64_3, builder.create<ConstantIntOp>(0x9FB21C651E98DF25ULL, builder.getI64Type()));

        // xorshift
        return builder.create<XOrIOp>(
            h64_4, builder.create<ShRUIOp>(h64_4, builder.create<ConstantIntOp>(28, builder.getI64Type())));
    }

    static Value XXH64_avalanche(ImplicitLocOpBuilder &builder, Value h64)
    {
        auto h64_1 = builder.create<XOrIOp>(
            h64, builder.create<ShRUIOp>(h64, builder.create<ConstantIntOp>(33, builder.getI64Type())));
        auto h64_2 =
            builder.create<MulIOp>(h64_1, builder.create<ConstantIntOp>(0xC2B2AE3D27D4EB4FULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            h64_2, builder.create<ShRUIOp>(h64_2, builder.create<ConstantIntOp>(28, builder.getI64Type())));
        auto h64_4 =
            builder.create<MulIOp>(h64_3, builder.create<ConstantIntOp>(0x165667B19E3779F9ULL, builder.getI64Type()));
        auto h64_5 = builder.create<XOrIOp>(
            h64_4, builder.create<ShRUIOp>(h64_4, builder.create<ConstantIntOp>(32, builder.getI64Type())));
        return h64_5;
    }

    static auto XXH3_len_1to3_64b(ImplicitLocOpBuilder &builder, ValueRange vals, const unsigned int len)
    {
        assert(vals.size() == 1);
        /*
         * len = 1: combined = { input[0], 0x01, input[0], input[0] }
         * len = 2: combined = { input[1], 0x02, input[0], input[1] }
         * len = 3: combined = { input[2], 0x03, input[0], input[1] }
         */
        {
            auto c1 = builder.create<TruncIOp>(
                builder.getI8Type(), builder.create<ShRUIOp>(vals[0], builder.create<ConstantIntOp>(
                                                                          (len - 1) * CHAR_BIT, vals[0].getType())));
            auto c2 = builder.create<TruncIOp>(
                builder.getI8Type(), builder.create<ShRUIOp>(vals[0], builder.create<ConstantIntOp>(
                                                                          (len >> 1) * CHAR_BIT, vals[0].getType())));
            auto c3 = builder.create<TruncIOp>(builder.getI8Type(), vals[0]);
            auto combined = builder.create<OrIOp>(
                builder.create<OrIOp>(
                    builder.create<OrIOp>(
                        builder.create<ShLIOp>(builder.create<ExtSIOp>(builder.getI32Type(), c1),
                                               builder.create<ConstantIntOp>(16, builder.getI32Type())),
                        builder.create<ShLIOp>(builder.create<ExtSIOp>(builder.getI32Type(), c2),
                                               builder.create<ConstantIntOp>(24, builder.getI32Type()))),
                    builder.create<ExtSIOp>(builder.getI32Type(), c3)),
                builder.create<ConstantIntOp>(len << 8, builder.getI32Type()));
            auto bitflip = builder.create<ConstantIntOp>(*reinterpret_cast<const uint32_t *>(XXH_SECRET.data()) ^
                                                             *reinterpret_cast<const uint32_t *>(XXH_SECRET.data() + 4),
                                                         builder.getI64Type());
            auto keyed = builder.create<XOrIOp>(builder.create<ExtSIOp>(builder.getI64Type(), combined), bitflip);
            return XXH64_avalanche(builder, keyed);
        }
    }

    static auto XXH3_len_4to8_64b(ImplicitLocOpBuilder &builder, ValueRange vals, const unsigned int len)
    {
        assert(vals.size() <= 2);
        /** TODO: allow also larger sizes with smaller than 4 byte size. However, in this case we have to do a
         * bunch of bitshifts in order to combine these values to a single 8 byte value
         **/
        Value input1, input2;
        if (vals.size() == 1) // split single 8 byte value
        {
            input1 = builder.create<TruncIOp>(builder.getI32Type(), vals[0]);
            input2 = builder.create<TruncIOp>(
                builder.getI32Type(),
                builder.create<ShRUIOp>(vals[0], builder.create<ConstantIntOp>(32, vals[0].getType())));
        }
        else
        {
            input1 = vals[0];
            input2 = vals[1];
        }

        auto bitflip = builder.create<XOrIOp>(
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 8),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 16),
                                          builder.getI64Type()));

        auto input64 =
            builder.create<AddIOp>(builder.create<ExtSIOp>(builder.getI64Type(), input2),
                                   builder.create<ShLIOp>(builder.create<ExtSIOp>(builder.getI64Type(), input1),
                                                          builder.create<ConstantIntOp>(32, builder.getI64Type())));
        auto keyed = builder.create<XOrIOp>(input64, bitflip);
        return XXH3_rrmxmx(builder, keyed, len);
    }

    static Value XXH3_len_9to16_64b(ImplicitLocOpBuilder &builder, ValueRange vals, const unsigned int len)
    {
        Value input1, input2;
        if (vals.size() == 2)
        {
            input1 = getINT<64>(builder, vals[0]);
            input2 = getINT<64>(builder, vals[1]);
        }
        else if (vals.size() == 4)
        {
            input1 = combine(builder, vals[0], vals[1]);
            input2 = combine(builder, vals[2], vals[3]);
        }
        else if (vals[0].getType().getIntOrFloatBitWidth() == 32 && vals[1].getType().getIntOrFloatBitWidth() == 32)
        {
            input1 = combine(builder, vals[0], vals[1]);
            input2 = getINT<64>(builder, vals[2]);
        }
        else if (vals[1].getType().getIntOrFloatBitWidth() == 32 && vals[2].getType().getIntOrFloatBitWidth() == 32)
        {
            input1 = getINT<64>(builder, vals[0]);
            input2 = combine(builder, vals[1], vals[2]);
        }
        else // vals[0].getType().getIntOrFloatBitWidth() == 32 && vals[1].getType().getIntOrFloatBitWidth() != 32
        {
            auto tmp = split(builder, vals[1]);
            input1 = combine(builder, vals[0], tmp.first);
            input2 = combine(builder, tmp.second, vals[2]);
        }

        auto bitflip1 = builder.create<XOrIOp>(
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 24),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 32),
                                          builder.getI64Type()));
        auto bitflip2 = builder.create<XOrIOp>(
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 40),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(*reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 48),
                                          builder.getI64Type()));
        const Value input_lo = builder.create<XOrIOp>(input1, bitflip1);
        const Value input_hi = builder.create<XOrIOp>(input2, bitflip2);
        auto acc = builder.create<AddIOp>(
            builder.create<AddIOp>(builder.create<AddIOp>(builder.create<ConstantIntOp>(len, builder.getI64Type()),
                                                          XXH_swap64(builder, input_lo)),
                                   input_hi),
            XXH3_mul128_fold64(builder, input_lo, input_hi));
        return XXH3_avalanche(builder, acc);
    }

    static Value XXH3_len_17to32_64b(ImplicitLocOpBuilder &builder, ValueRange vals, const unsigned int len)
    {
        Value acc = builder.create<ConstantIntOp>(len * XXH_PRIME64_1, 64);
        llvm::SmallVector<Value, 4> v64;
        auto iter = vals.begin();
        while (iter != vals.end())
        {
            if ((*iter).getType().getIntOrFloatBitWidth() == 64)
            {
                v64.push_back(*iter);
                ++iter;
            }
            else if (iter + 1 == vals.end())
            {
                v64.push_back(builder.create<ExtUIOp>(builder.getI64Type(), *iter));
                ++iter;
            }
            else
            {
                if ((*(iter + 1)).getType().getIntOrFloatBitWidth() == 32)
                {
                    auto upper = builder.create<ExtUIOp>(builder.getI64Type(), *iter++);
                    v64.push_back(
                        builder.create<OrIOp>(builder.create<ShLIOp>(upper, builder.create<ConstantIntOp>(32, 64)),
                                              builder.create<ExtUIOp>(builder.getI64Type(), *iter)));
                    ++iter;
                }
                else
                {
                    auto upper = builder.create<ExtUIOp>(builder.getI64Type(), *iter++);
                    auto sp = split(builder, *(iter));
                    v64.push_back(builder.create<OrIOp>(
                        builder.create<ShLIOp>(upper, builder.create<ConstantIntOp>(32, 64)), sp.first));
                    *iter = sp.second;
                }
            }
        }

        if (len < 32)
        {
            v64.push_back(builder.create<ConstantIntOp>(0, 64));
        }

        acc = builder.create<AddIOp>(acc, XXH3_mix16B(builder, {v64[0], v64[1]}, 0));
        acc = builder.create<AddIOp>(acc, XXH3_mix16B(builder, {v64[2], v64[3]}, 16));
        return XXH3_avalanche(builder, acc);
    }

    static auto hashFunc(ImplicitLocOpBuilder &builder, ValueRange vals)
    {
        SmallVector<Value> intVals;
        unsigned int size = 0;
        for (const auto &val : vals.drop_back())
        {
            const auto &type = val.getType();
            if (type.isIntOrFloat())
            {
                intVals.push_back(builder.create<BitcastOp>(builder.getIntegerType(type.getIntOrFloatBitWidth()), val));
            }
            else if (type.isIndex())
            {
                intVals.push_back(builder.create<IndexCastOp>(builder.getI64Type(), val));
            }
            else
            {
                throw std::logic_error("Type is currently not supported");
            }
            size += intVals.back().getType().getIntOrFloatBitWidth() / CHAR_BIT;
        }

        SmallVector<Value, 1> res;

        assert(size != 0);

        if (size <= 4)
        {
            res.push_back(XXH3_len_1to3_64b(builder, intVals, size));
        }
        else if (size <= 8)
        {
            res.push_back(XXH3_len_4to8_64b(builder, intVals, size));
        }
        else if (size <= 16)
        {
            res.push_back(XXH3_len_9to16_64b(builder, intVals, size));
        }
        else if (size <= 32)
        {
            res.push_back(XXH3_len_17to32_64b(builder, intVals, size));
        }
        else
        {
            throw NotImplementedException("Hash func not implemented for size larger than 32 byte");
        }

        return res;
    };

    LogicalResult
    HashOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        auto hOp = dyn_cast<HashOp>(op);
        HashOpAdaptor hashOpAdaptor(hOp);
        auto loc = op->getLoc();
        ImplicitLocOpBuilder builder(loc, rewriter);

        const auto &shape = hashOpAdaptor.getInput().front().getType().dyn_cast<::mlir::TensorType>().getShape();
        for (const auto &in : hashOpAdaptor.getInput())
        {
            assert(in.getType().isa<TensorType>());
            assert(in.getType().dyn_cast<TensorType>().getShape() == shape);
        }

        ::mlir::Value outTensor;
        if (hOp.getResult().getType().dyn_cast<ShapedType>().hasStaticShape())
        {
            outTensor = builder.create<tensor::EmptyOp>(
                hOp.getResult().getType().dyn_cast<ShapedType>().getShape(), builder.getI64Type());
        }
        else
        {
            SmallVector<Value, 1> outTensorSize;
            outTensorSize.push_back(builder.create<tensor::DimOp>(hashOpAdaptor.getInput().front(), 0));
            outTensor = builder.create<tensor::EmptyOp>( llvm::makeArrayRef<int64_t>(-1), builder.getI64Type(),outTensorSize);
        }

        SmallVector<Value, 1> res;
        res.push_back(outTensor);

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getParallelIteratorTypeName());

        SmallVector<Type, 1> ret_type;
        ret_type.push_back(outTensor.getType());
        SmallVector<AffineMap> indexing_maps(hashOpAdaptor.getInput().size() + 1 + bool(hashOpAdaptor.getPred()),
                                             builder.getDimIdentityMap());

        SmallVector<Value> inputs;
        if (hashOpAdaptor.getPred())
            inputs.push_back(hashOpAdaptor.getPred());
        inputs.insert(inputs.end(), hashOpAdaptor.getInput().begin(), hashOpAdaptor.getInput().end());

        auto linalgOp = builder.create<linalg::GenericOp>(
            /*results*/ ret_type,
            /*inputs*/ inputs, /*outputs*/ res,
            /*indexing maps*/ indexing_maps,
            /*iterator types*/ iter_type,
            [&hashOpAdaptor](OpBuilder &nestedBuilder, Location loc, ValueRange vals)
            {
                ImplicitLocOpBuilder builder(loc, nestedBuilder);

                if (hashOpAdaptor.getPred())
                {
                    auto pred = vals.take_front()[0];
                    vals = vals.drop_front();
                    auto res =
                        builder
                            .create<scf::IfOp>(
                                builder.getI64Type(), pred,
                                [&](OpBuilder &b, Location loc)
                                {
                                    ImplicitLocOpBuilder nb(loc, b);
                                    auto res = hashFunc(nb, vals);
                                    b.create<scf::YieldOp>(loc, res);
                                },
                                [&](OpBuilder &b, Location loc) {
                                    b.create<scf::YieldOp>(loc, llvm::makeArrayRef<Value>(
                                                                    b.create<ConstantIntOp>(loc, 0, b.getI64Type())));
                                })
                            ->getResults();
                    builder.create<linalg::YieldOp>(res);
                }
                else
                {
                    builder.create<linalg::YieldOp>(hashFunc(builder, vals));
                }
            });

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }
} // namespace voila::mlir::lowering