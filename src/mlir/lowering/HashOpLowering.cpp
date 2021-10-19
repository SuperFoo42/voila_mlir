#include "mlir/lowering/HashOpLowering.hpp"

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using ::mlir::voila::HashOp;
    using ::mlir::voila::HashOpAdaptor;

    HashOpLowering::HashOpLowering(MLIRContext *ctx) : ConversionPattern(HashOp::getOperationName(), 1, ctx) {}

    static constexpr auto XXH_SECRET_DEFAULT_SIZE = 192;

    static constexpr std::array<uint8_t, XXH_SECRET_DEFAULT_SIZE> XXH_SECRET = {
        0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4,
        0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f, 0xcb, 0x79, 0xe6, 0x4e,
        0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43,
        0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c, 0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb,
        0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3, 0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19,
        0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8, 0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7,
        0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78,
        0x73, 0x64, 0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
        0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e, 0x2b, 0x16,
        0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f,
        0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e};

    // TODO: check if this becomes a bswap instruction when compiled
    static Value XXH_swap64(OpBuilder &builder, const Location &loc, Value x)
    {
        assert(x.getType().getIntOrFloatBitWidth() == 64);

        return builder.create<OrIOp>(
            loc,
            builder.create<OrIOp>(
                loc,
                builder.create<OrIOp>(
                    loc,
                    builder.create<OrIOp>(
                        loc,
                        builder.create<OrIOp>(
                            loc,
                            builder.create<OrIOp>(
                                loc,
                                builder.create<OrIOp>(
                                    loc,
                                    builder.create<AndIOp>(
                                        loc,
                                        builder.create<ShLIOp>(
                                            loc, x, builder.create<ConstantIntOp>(loc, 56, builder.getI64Type())),
                                        builder.create<ConstantIntOp>(loc, 0xff00000000000000ULL,
                                                                      builder.getI64Type())),
                                    builder.create<AndIOp>(
                                        loc,
                                        builder.create<ShLIOp>(
                                            loc, x, builder.create<ConstantIntOp>(loc, 40, builder.getI64Type())),
                                        builder.create<ConstantIntOp>(loc, 0x00ff000000000000ULL,
                                                                      builder.getI64Type()))),
                                builder.create<AndIOp>(
                                    loc,
                                    builder.create<ShLIOp>(
                                        loc, x, builder.create<ConstantIntOp>(loc, 24, builder.getI64Type())),
                                    builder.create<ConstantIntOp>(loc, 0x0000ff0000000000ULL, builder.getI64Type()))),
                            builder.create<AndIOp>(
                                loc,
                                builder.create<ShLIOp>(loc, x,
                                                       builder.create<ConstantIntOp>(loc, 8, builder.getI64Type())),
                                builder.create<ConstantIntOp>(loc, 0x000000ff00000000ULL, builder.getI64Type()))),
                        builder.create<AndIOp>(
                            loc,
                            builder.create<ShRUIOp>(loc, x,
                                                    builder.create<ConstantIntOp>(loc, 8, builder.getI64Type())),
                            builder.create<ConstantIntOp>(loc, 0x00000000ff000000ULL, builder.getI64Type()))),
                    builder.create<AndIOp>(
                        loc,
                        builder.create<ShRUIOp>(loc, x, builder.create<ConstantIntOp>(loc, 24, builder.getI64Type())),
                        builder.create<ConstantIntOp>(loc, 0x0000000000ff0000ULL, builder.getI64Type()))),
                builder.create<AndIOp>(
                    loc, builder.create<ShRUIOp>(loc, x, builder.create<ConstantIntOp>(loc, 40, builder.getI64Type())),
                    builder.create<ConstantIntOp>(loc, 0x000000000000ff00ULL, builder.getI64Type()))),
            builder.create<AndIOp>(
                loc, builder.create<ShRUIOp>(loc, x, builder.create<ConstantIntOp>(loc, 56, builder.getI64Type())),
                builder.create<ConstantIntOp>(loc, 0x00000000000000ffULL, builder.getI64Type())));
    }

    template<int WIDTH>
    static Value getINT(OpBuilder &builder, const Location &loc, Value val)
    {
        if (val.getType().getIntOrFloatBitWidth() < WIDTH)
        {
            return builder.create<ExtSIOp>(loc, val, builder.getIntegerType(WIDTH));
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

    static Value XXH3_mul128_fold64(OpBuilder &builder, const Location loc, Value lhs, Value rhs)
    {
        auto lhs128 = builder.create<ExtSIOp>(loc, lhs, builder.getIntegerType(128));
        auto rhs128 = builder.create<ExtSIOp>(loc, rhs, builder.getIntegerType(128));
        auto product = builder.create<MulIOp>(loc, lhs128, rhs128);
        auto product_low = builder.create<TruncIOp>(loc, product, builder.getI64Type());
        auto product_high = builder.create<TruncIOp>(
            loc, builder.create<ShRUIOp>(loc, product, builder.create<ConstantIntOp>(loc, 64, product.getType())),
            builder.getI64Type());

        return builder.create<XOrIOp>(loc, product_low, product_high);
    }

    static Value XXH3_avalanche(OpBuilder &builder, const Location loc, Value h64)
    {
        auto h64_1 = builder.create<XOrIOp>(
            loc, h64, builder.create<ShRUIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 37, builder.getI64Type())));
        auto h64_2 = builder.create<MulIOp>(
            loc, h64_1, builder.create<ConstantIntOp>(loc, 0x165667919E3779F9ULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            loc, h64_2,
            builder.create<ShRUIOp>(loc, h64_2, builder.create<ConstantIntOp>(loc, 32, builder.getI64Type())));
        return h64_3;
    }

    static Value XXH3_rrmxmx(OpBuilder &builder, const Location loc, Value h64, unsigned int len)
    {
        // rotate h64
        // TODO: check that this become a rotl instruction
        auto h64_r1 = builder.create<OrIOp>(
            loc, builder.create<ShLIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 49, builder.getI64Type())),
            builder.create<ShRUIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 15, builder.getI64Type())));
        auto h64_r2 = builder.create<OrIOp>(
            loc, builder.create<ShLIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 24, builder.getI64Type())),
            builder.create<ShRUIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 40, builder.getI64Type())));
        auto h64_1 = builder.create<XOrIOp>(loc, h64, builder.create<XOrIOp>(loc, h64_r1, h64_r2));
        auto h64_2 = builder.create<MulIOp>(
            loc, h64_1, builder.create<ConstantIntOp>(loc, 0x9FB21C651E98DF25ULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            loc, h64_2,
            builder.create<AddIOp>(
                loc, builder.create<ShRUIOp>(loc, h64_2, builder.create<ConstantIntOp>(loc, 35, builder.getI64Type())),
                builder.create<ConstantIntOp>(loc, len, builder.getI64Type())));
        auto h64_4 = builder.create<MulIOp>(
            loc, h64_3, builder.create<ConstantIntOp>(loc, 0x9FB21C651E98DF25ULL, builder.getI64Type()));

        // xorshift
        return builder.create<XOrIOp>(
            loc, h64_4,
            builder.create<ShRUIOp>(loc, h64_4, builder.create<ConstantIntOp>(loc, 28, builder.getI64Type())));
    }

    static Value XXH64_avalanche(OpBuilder &builder, const Location loc, Value h64)
    {
        auto h64_1 = builder.create<XOrIOp>(
            loc, h64, builder.create<ShRUIOp>(loc, h64, builder.create<ConstantIntOp>(loc, 33, builder.getI64Type())));
        auto h64_2 = builder.create<MulIOp>(
            loc, h64_1, builder.create<ConstantIntOp>(loc, 0xC2B2AE3D27D4EB4FULL, builder.getI64Type()));
        auto h64_3 = builder.create<XOrIOp>(
            loc, h64_2,
            builder.create<ShRUIOp>(loc, h64_2, builder.create<ConstantIntOp>(loc, 28, builder.getI64Type())));
        auto h64_4 = builder.create<MulIOp>(
            loc, h64_3, builder.create<ConstantIntOp>(loc, 0x165667B19E3779F9ULL, builder.getI64Type()));
        auto h64_5 = builder.create<XOrIOp>(
            loc, h64_4,
            builder.create<ShRUIOp>(loc, h64_4, builder.create<ConstantIntOp>(loc, 32, builder.getI64Type())));
        return h64_5;
    }

    static auto XXH3_len_1to3_64b(OpBuilder &builder, const Location loc, ValueRange vals, const unsigned int len)
    {
        assert(vals.size() == 1);
        /*
         * len = 1: combined = { input[0], 0x01, input[0], input[0] }
         * len = 2: combined = { input[1], 0x02, input[0], input[1] }
         * len = 3: combined = { input[2], 0x03, input[0], input[1] }
         */
        {
            auto c1 = builder.create<TruncIOp>(
                loc,
                builder.create<ShRUIOp>(loc, vals[0],
                                        builder.create<ConstantIntOp>(loc, (len - 1) * CHAR_BIT, vals[0].getType())),
                builder.getI8Type());
            auto c2 = builder.create<TruncIOp>(
                loc,
                builder.create<ShRUIOp>(loc, vals[0],
                                        builder.create<ConstantIntOp>(loc, (len >> 1) * CHAR_BIT, vals[0].getType())),
                builder.getI8Type());
            auto c3 = builder.create<TruncIOp>(loc, vals[0], builder.getI8Type());
            auto combined = builder.create<OrIOp>(
                loc,
                builder.create<OrIOp>(
                    loc,
                    builder.create<OrIOp>(
                        loc,
                        builder.create<ShLIOp>(loc, builder.create<ExtSIOp>(loc, c1, builder.getI32Type()),
                                               builder.create<ConstantIntOp>(loc, 16, builder.getI32Type())),
                        builder.create<ShLIOp>(loc, builder.create<ExtSIOp>(loc, c2, builder.getI32Type()),
                                               builder.create<ConstantIntOp>(loc, 24, builder.getI32Type()))),
                    builder.create<ExtSIOp>(loc, c3, builder.getI32Type())),
                builder.create<ConstantIntOp>(loc, len << 8, builder.getI32Type()));
            auto bitflip = builder.create<ConstantIntOp>(loc,
                                                         *reinterpret_cast<const uint32_t *>(XXH_SECRET.data()) ^
                                                             *reinterpret_cast<const uint32_t *>(XXH_SECRET.data() + 4),
                                                         builder.getI64Type());
            auto keyed =
                builder.create<XOrIOp>(loc, builder.create<ExtSIOp>(loc, combined, builder.getI64Type()), bitflip);
            return XXH64_avalanche(builder, loc, keyed);
        }
    }

    static auto XXH3_len_4to8_64b(OpBuilder &builder, const Location loc, ValueRange vals, const unsigned int len)
    {
        assert(vals.size() <= 2);
        /** TODO: allow also larger sizes with smaller than 4 byte size. However, in this case we have to do a
         * bunch of bitshifts in order to combine these values to a single 8 byte value
         **/
        Value input1, input2;
        if (vals.size() == 1) // split single 8 byte value
        {
            input1 = builder.create<TruncIOp>(loc, vals[0], builder.getI32Type());
            input2 = builder.create<TruncIOp>(
                loc, builder.create<ShRUIOp>(loc, vals[0], builder.create<ConstantIntOp>(loc, 32, vals[0].getType())),
                builder.getI32Type());
        }
        else
        {
            input1 = vals[0];
            input2 = vals[1];
        }

        auto bitflip = builder.create<XOrIOp>(
            loc,
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 8),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 16),
                                          builder.getI64Type()));

        auto input64 = builder.create<AddIOp>(
            loc, builder.create<ExtSIOp>(loc, input2, builder.getI64Type()),
            builder.create<ShLIOp>(loc, builder.create<ExtSIOp>(loc, input1, builder.getI64Type()),
                                   builder.create<ConstantIntOp>(loc, 32, builder.getI64Type())));
        auto keyed = builder.create<XOrIOp>(loc, input64, bitflip);
        return XXH3_rrmxmx(builder, loc, keyed, len);
    }

    static Value XXH3_len_9to16_64b(OpBuilder &builder, const Location loc, ValueRange vals, const unsigned int len)
    {
        assert(vals.size() == 2);
        /** TODO: allow also larger sizes with smaller than 4 byte size. However, in this case we have to do a
         * bunch of bitshifts in order to combine these values to a single 8 byte value
         **/
        auto input1 = getINT<64>(builder, loc, vals[0]);
        auto input2 = getINT<64>(builder, loc, vals[1]);

        auto bitflip1 = builder.create<XOrIOp>(
            loc,
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 24),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 32),
                                          builder.getI64Type()));
        auto bitflip2 = builder.create<XOrIOp>(
            loc,
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 40),
                                          builder.getI64Type()),
            builder.create<ConstantIntOp>(loc, *reinterpret_cast<const uint64_t *>(XXH_SECRET.data() + 48),
                                          builder.getI64Type()));
        const Value input_lo = builder.create<XOrIOp>(loc, input1, bitflip1);
        const Value input_hi = builder.create<XOrIOp>(loc, input2, bitflip2);
        auto acc = builder.create<AddIOp>(
            loc,
            builder.create<AddIOp>(loc,
                                   builder.create<AddIOp>(loc,
                                                          builder.create<ConstantIntOp>(loc, len, builder.getI64Type()),
                                                          XXH_swap64(builder, loc, input_lo)),
                                   input_hi),
            XXH3_mul128_fold64(builder, loc, input_lo, input_hi));
        return XXH3_avalanche(builder, loc, acc);
    }

    /* TODO
     * static auto mediumHashFunc(OpBuilder &builder, Location loc, ValueRange vals)

        {
            return;
        }
        */

    static auto hashFunc(OpBuilder &builder, Location loc, ValueRange vals)
    {
        SmallVector<Value> intVals;
        unsigned int size = 0;
        for (size_t i = 0; i < vals.size() - 1; ++i)
        {
            if (vals[i].getType().isIntOrFloat())
            {
                intVals.push_back(builder.create<BitcastOp>(
                    loc, vals[i], builder.getIntegerType(vals[i].getType().getIntOrFloatBitWidth())));
            }
            else if (vals[i].getType().isIndex())
            {
                intVals.push_back(builder.create<IndexCastOp>(loc, vals[i], builder.getI64Type()));
            }
            else
            {
                throw std::logic_error("Type can not be hashed");
            }
            size += intVals.back().getType().getIntOrFloatBitWidth() / CHAR_BIT;
        }

        SmallVector<Value, 1> res;

        assert(size != 0);

        if (size <= 4)
        {
            res.push_back(XXH3_len_1to3_64b(builder, loc, intVals, size));
        }
        else if (size <= 8)
        {
            res.push_back(XXH3_len_4to8_64b(builder, loc, intVals, size));
        }
        else if (size <= 16)
        {
            res.push_back(XXH3_len_9to16_64b(builder, loc, intVals, size));
        }

        /*TODO
         * else if (size <= 128)
            res.push_back(mediumHashFunc(builder, loc, vals, size));
            */
        else
        {
            throw std::logic_error("Hash func not implemented for size larger than 128 byte");
        }

        builder.create<linalg::YieldOp>(loc, res);
    };

    LogicalResult
    HashOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const
    {
        HashOpAdaptor hashOpAdaptor(operands);
        auto loc = op->getLoc();
        // TODO: murmur3 for strings
        const auto &shape = hashOpAdaptor.input().front().getType().dyn_cast<::mlir::TensorType>().getShape();
        for (const auto &in : hashOpAdaptor.input())
        {
            assert(in.getType().isa<TensorType>());
            assert(in.getType().dyn_cast<TensorType>().getShape() == shape);
        }

        ::mlir::Value outTensor;
        if (hashOpAdaptor.input().getType().front().dyn_cast<TensorType>().hasStaticShape())
        {
            outTensor = rewriter.create<linalg::InitTensorOp>(loc, shape, rewriter.getI64Type());
        }
        else
        {
            SmallVector<Value, 1> outTensorSize;
            outTensorSize.push_back(rewriter.create<tensor::DimOp>(loc, hashOpAdaptor.input().front(), 0));
            outTensor = rewriter.create<linalg::InitTensorOp>(loc, outTensorSize, rewriter.getI64Type());
        }

        SmallVector<Value, 1> res;
        res.push_back(outTensor);

        SmallVector<StringRef, 1> iter_type;
        iter_type.push_back(getParallelIteratorTypeName());

        SmallVector<Type, 1> ret_type;
        ret_type.push_back(outTensor.getType());
        SmallVector<AffineMap> indexing_maps(hashOpAdaptor.input().size() + 1, rewriter.getDimIdentityMap());

        auto linalgOp = rewriter.create<linalg::GenericOp>(loc, /*results*/ ret_type,
                                                           /*inputs*/ hashOpAdaptor.input(), /*outputs*/ res,
                                                           /*indexing maps*/ indexing_maps,
                                                           /*iterator types*/ iter_type, hashFunc);

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }
} // namespace voila::mlir::lowering