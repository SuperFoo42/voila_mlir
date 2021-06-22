#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/VoilaDialect.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/Sequence.h"

#include <MLIRLoweringError.hpp>

namespace voila::mlir::lowering
{
    struct ReadOpLowering : public ::mlir::ConversionPattern
    {
        explicit ReadOpLowering(::mlir::MLIRContext *ctx) :
            ConversionPattern(::mlir::voila::ReadOp::getOperationName(), 1, ctx)
        {
        }

        static ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
        {
            assert(type.hasRank() && "expected only ranked shapes");
            return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
        }

        ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op,
                                              ::mlir::ArrayRef<::mlir::Value> operands,
                                              ::mlir::ConversionPatternRewriter &rewriter) const final
        {
            ::mlir::voila::ReadOpAdaptor readOpAdaptor(operands);
            ::mlir::SmallVector<::mlir::Value> ops;
            auto loc = op->getLoc();
            for (auto o : op->getOperands())
            {
                ops.push_back(o);
            }

            ::mlir::Value col;
            //TODO: only for tensors
            if (readOpAdaptor.column().getType().isa<::mlir::TensorType>())
            {
                col = rewriter.create<::mlir::memref::BufferCastOp>(
                    loc, convertTensorToMemRef(readOpAdaptor.column().getType().dyn_cast<::mlir::TensorType>()),
                    readOpAdaptor.column());
            }
            else if (readOpAdaptor.column().getType().isa<::mlir::MemRefType>())
            {
                col = readOpAdaptor.column();
            }
            else
            {
                throw ::voila::MLIRLoweringError();
            }

            ::mlir::SmallVector<::mlir::Value> sizes, offsets, strides;
            strides.push_back(rewriter.create<::mlir::ConstantIndexOp>(loc, 1));
            offsets.push_back(
                rewriter.create<::mlir::IndexCastOp>(loc, readOpAdaptor.index(), rewriter.getIndexType()));
            sizes.push_back(rewriter.create<::mlir::memref::DimOp>(loc, col, 0));

            rewriter.replaceOpWithNewOp<::mlir::memref::SubViewOp>(op, col, offsets, sizes, strides);
            return ::mlir::success();
        }
    };
} // namespace voila::mlir::lowering