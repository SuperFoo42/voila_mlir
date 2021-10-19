#include "mlir/VoilaOps.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

#define GET_OP_CLASSES
#include "mlir/VoilaOps.cpp.inc"
using namespace ::mlir::arith;
mlir::TensorType static inferShapesFromBinaryOp(mlir::Type lhsType, mlir::Type rhsType)
{
    int64_t dimSize = 1;
    mlir::Type elemType;
    if (lhsType.isa<mlir::TensorType>() && lhsType.dyn_cast<mlir::TensorType>().getElementType().isa<mlir::FloatType>())
    {
        elemType = lhsType.dyn_cast<mlir::TensorType>().getElementType();
    }

    else if (lhsType.isa<mlir::FloatType>())
    {
        elemType = lhsType;
    }
    else if (rhsType.isa<mlir::TensorType>())
    {
        elemType = rhsType.dyn_cast<mlir::TensorType>().getElementType();
    }
    else
    {
        elemType = rhsType;
    }

    if (lhsType.isa<mlir::TensorType>())
    {
        dimSize = lhsType.dyn_cast<mlir::TensorType>().getDimSize(0);
    }
    if (rhsType.isa<mlir::TensorType>())
    {
        dimSize = std::max(dimSize, rhsType.dyn_cast<mlir::TensorType>().getDimSize(0));
    }

    return mlir::RankedTensorType::get(dimSize, elemType);
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
mlir::CallInterfaceCallable mlir::voila::GenericCallOp::getCallableForCallee()
{
    return callee();
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
mlir::Operation::operand_range mlir::voila::GenericCallOp::getArgOperands()
{
    return inputs();
}

bool mlir::voila::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
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

::mlir::LogicalResult mlir::voila::SelectOp::canonicalize(SelectOp op, PatternRewriter &rewriter)
{
    SmallVector<std::reference_wrapper<OpOperand>> uses;
    for (auto &use : op->getUses())
        uses.push_back(use);
    while (!uses.empty())
    {
        OpOperand &use = uses.pop_back_val();
        Operation *user = use.getOwner();
        // test if replacement with select operation would produce unsafe results
        if (::mlir::isa<mlir::voila::InsertOp>(user) || ::mlir::isa<mlir::voila::EmitOp>(user) ||
            ::mlir::isa<mlir::voila::HashOp>(user) || ::mlir::isa<mlir::voila::ReadOp>(user) ||
            ::mlir::isa<mlir::voila::WriteOp>(user) || ::mlir::isa<mlir::voila::LookupOp>(user) ||
            ::mlir::isa<mlir::voila::EqOp>(user) || ::mlir::isa<mlir::voila::GeOp>(user) ||
            ::mlir::isa<mlir::voila::GeOp>(user) || ::mlir::isa<mlir::voila::GeqOp>(user) ||
            ::mlir::isa<mlir::voila::LeOp>(user) || ::mlir::isa<mlir::voila::LeqOp>(user) ||
            ::mlir::isa<mlir::voila::NeqOp>(user) || ::mlir::isa<mlir::voila::AvgOp>(user))
        {
            return failure();
        }
        else if (::mlir::isa<mlir::voila::SumOp>(user) || ::mlir::isa<mlir::voila::CountOp>(user) ||
                 ::mlir::isa<mlir::voila::MinOp>(user) || ::mlir::isa<mlir::voila::MaxOp>(user))
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
}