#include "mlir/IR/VoilaOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#define GET_OP_CLASSES
#include "mlir/IR/VoilaOps.cpp.inc"
using namespace ::mlir;
using namespace ::mlir::arith;
using namespace mlir::voila;
/// Return the callee of the generic call operation, this is required by the
/// call interface.
[[maybe_unused]] CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
    return callee();
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
[[maybe_unused]] Operation::operand_range GenericCallOp::getArgOperands()
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

LogicalResult mlir::voila::SelectOp::canonicalize(mlir::voila::SelectOp op, PatternRewriter &rewriter)
{
    SmallVector<std::reference_wrapper<OpOperand>> uses;
    SmallVector<Value> toPredicate;
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
        else if (isa<SumOp>(user))
        {
            if (dyn_cast<SumOp>(user).indices() == Value())
                continue;
            else
                return failure();
        }
        else if (isa<CountOp>(user))
        {
            if (dyn_cast<CountOp>(user).indices() == Value())
                continue;
            else
                return failure();
        }
        else if (isa<MinOp>(user))
        {
            if (dyn_cast<MinOp>(user).indices() == Value())
                continue;
            else
                return failure();
        }
        else if (isa<MaxOp>(user))
        {
            if (dyn_cast<MaxOp>(user).indices() == Value())
                continue;
            else
                return failure();
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

void AddOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool AddOp::predicated()
{
    return pred() == Value();
}

void NeqOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool NeqOp::predicated()
{
    return pred() == Value();
}

void MulOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool MulOp::predicated()
{
    return pred() == Value();
}

void ModOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool ModOp::predicated()
{
    return pred() == Value();
}

void LeqOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool LeqOp::predicated()
{
    return pred() == Value();
}

void LeOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool LeOp::predicated()
{
    return pred() == Value();
}

void GeqOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool GeqOp::predicated()
{
    return pred() == Value();
}

void GeOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool GeOp::predicated()
{
    return pred() == Value();
}

void EqOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool EqOp::predicated()
{
    return pred() == Value();
}

void DivOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool DivOp::predicated()
{
    return pred() == Value();
}

void AndOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool AndOp::predicated()
{
    return pred() == Value();
}

void SubOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool SubOp::predicated()
{
    return pred() == Value();
}

void OrOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool OrOp::predicated()
{
    return pred() == Value();
}

void NotOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool NotOp::predicated()
{
    return pred() == Value();
}

void ScatterOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool ScatterOp::predicated()
{
    return pred() == Value();
}

void ReadOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool ReadOp::predicated()
{
    return pred() == Value();
}

void MinOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool MinOp::predicated()
{
    return pred() == Value();
}

void MaxOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool MaxOp::predicated()
{
    return pred() == Value();
}

void LookupOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool LookupOp::predicated()
{
    return pred() == Value();
}

void InsertOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool InsertOp::predicated()
{
    return pred() == Value();
}

void HashOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool HashOp::predicated()
{
    return pred() == Value();
}

void GatherOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool GatherOp::predicated()
{
    return pred() == Value();
}

void CountOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool CountOp::predicated()
{
    return pred() == Value();
}

void AvgOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool AvgOp::predicated()
{
    return pred() == Value();
}

void WriteOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool WriteOp::predicated()
{
    return pred() == Value();
}

void SumOp::predicate(Value pred)
{
    if (predicated())
        throw; //TODO
    predMutable().assign(pred);
}

bool SumOp::predicated()
{
    return pred() == Value();
}