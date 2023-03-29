#include "mlir/Dialects/Voila/lowering/ReadOpLowering.hpp"
#include "MLIRLoweringError.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/lowering/utility/TypeUtils.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include <cassert>

namespace voila::mlir::lowering
{
    using namespace ::mlir;
    using namespace ::mlir::arith;
    using namespace ::mlir::bufferization;
    using ::mlir::voila::ReadOp;
    using ::mlir::voila::ReadOpAdaptor;

    LogicalResult ReadOpLowering::matchAndRewrite(::mlir::voila::ReadOp op,
                                                  OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
    {
        auto loc = op->getLoc();

        Value col;
        // TODO: only for tensors
        if (isTensor(op.getColumn()))
        {
            col = rewriter.create<ToMemrefOp>(
                loc, convertTensorToMemRef(asTensorType(op.getColumn())), op.getColumn());
        }
        else if (isMemRef(op.getColumn()))
        {
            col = op.getColumn();
        }
        else
        {
            throw MLIRLoweringError();
        }

        SmallVector<Value> sizes, offsets, strides;
        strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
        offsets.push_back(rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), op.getIndex()));
        sizes.push_back(rewriter.create<memref::DimOp>(loc, col, 0));

        rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, col, offsets, sizes, strides);
        return success();
    }
} // namespace voila::mlir::lowering