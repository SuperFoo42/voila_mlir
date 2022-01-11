#include "mlir/IR/VoilaOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/LoopUtils.h"

#define GET_OP_CLASSES
#include "mlir/IR/VoilaOps.cpp.inc"
using namespace ::mlir;
using namespace ::mlir::arith;
using namespace mlir::voila;
/// Return the callee of the generic call operation, this is required by the
/// call interface.
[[maybe_unused]] mlir::CallInterfaceCallable mlir::voila::GenericCallOp::getCallableForCallee()
{
    return callee();
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
[[maybe_unused]] mlir::Operation::operand_range mlir::voila::GenericCallOp::getArgOperands()
{
    return inputs();
}

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    // The inputs must be Tensors with the same element type.
    auto input = inputs.front().dyn_cast<TensorType>();
    auto output = outputs.front().dyn_cast<TensorType>();
    if (!input || !output || input.getElementType() != output.getElementType())
        return false;
    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
}
/*

LogicalResult mlir::voila::SelectOp::canonicalize(mlir::voila::SelectOp op, PatternRewriter &rewriter)
{
    SmallVector<std::reference_wrapper<OpOperand>> uses;
    for (auto &use : op->getUses())
        uses.push_back(use);
    while (!uses.empty())
    {
        OpOperand &use = uses.pop_back_val();
        Operation *user = use.getOwner();
        // test if replacement with select operation would produce unsafe results
        if (isa<InsertOp>(user) || isa<EmitOp>(user) || isa<HashOp>(user) || isa<ReadOp>(user) || isa<WriteOp>(user) ||
            isa<LookupOp>(user) || isa<EqOp>(user) || isa<GeOp>(user) || isa<GeOp>(user) || isa<GeqOp>(user) ||
            isa<LeOp>(user) || isa<LeqOp>(user) || isa<NeqOp>(user) || isa<AvgOp>(user))
        {
            return failure();
        }
        else if (isa<SumOp>(user) || isa<CountOp>(user) || isa<MinOp>(user) || isa<MaxOp>(user))
        {
            continue;
        }
        for (auto &u : user->getUses())
            uses.push_back(u);
    }

    auto loc = op->getLoc();
    Value falseSel;
    Value tmp;
    if (op.values().getType().dyn_cast<TensorType>().hasStaticShape())
    {
        tmp = rewriter.create<linalg::InitTensorOp>(loc, op.values().getType().dyn_cast<TensorType>().getShape(),
                                                    getElementTypeOrSelf(op.values()));
    }
    else
    {
        auto dimShape = rewriter.create<tensor::DimOp>(loc, op.values(), 0).result();
        tmp = rewriter.create<linalg::InitTensorOp>(loc, ::llvm::makeArrayRef(dimShape),
                                                    getElementTypeOrSelf(op.values()));
    };

    if (getElementTypeOrSelf(op.values()).isa<IntegerType>())
    {
        falseSel = rewriter
                       .create<linalg::FillOp>(loc,
                                               rewriter.create<ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(),
                                                                              rewriter.getI64Type()),
                                               tmp)
                       .result();
    }
    else if (getElementTypeOrSelf(op.values()).isa<FloatType>())
    {
        falseSel = rewriter
                       .create<linalg::FillOp>(loc,
                                               rewriter.create<ConstantFloatOp>(
                                                   loc, rewriter.getF64FloatAttr(0).getValue(), rewriter.getF64Type()),
                                               tmp)
                       .result();
    }
    else
    {
        throw std::logic_error("Invalid type"); // TODO
    }

    rewriter.replaceOpWithNewOp<::mlir::SelectOp>(op, op.pred(), op.values(), falseSel);

    return success();
}*/
