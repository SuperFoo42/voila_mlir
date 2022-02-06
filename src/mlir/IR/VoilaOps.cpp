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
#include "NotImplementedException.hpp"
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
   /* SmallVector<OpOperand> uses;
    SmallVector<Value> toPredicate;

    auto loc = op->getLoc();

    //TODO: transitive traverse uses and look if we want to materialize this selection or use select

    //TODO: otherwise, predicate direct uses

    //TODO: do we even need this, if fusion works correctly?

    for (auto &use : op->getUses())
        uses.push_back(use);
    while (!uses.empty())
    {
        OpOperand use = uses.pop_back_val();
        auto *user = use.getOwner();
        // test if replacement with select operation would produce unsafe results
        if (isa<InsertOp>(user) || isa<HashOp>(user) || isa<ReadOp>(user) || isa<WriteOp>(user) ||
            isa<LookupOp>(user) || isa<EqOp>(user) || isa<GeOp>(user) || isa<GeOp>(user) || isa<GeqOp>(user) ||
            isa<LeOp>(user) || isa<LeqOp>(user) || isa<NeqOp>(user))
        {
            llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.pred());
            user->replaceUsesOfWith(use.get(), op.values());
             //use.drop();
            continue;
        }
        else if (isa<AddOp>(user))
        {
            //llvm::dyn_cast<PredicationOpInterface>(user).predicate(op.pred());
            auto operands = user->getOperands();
            operands[use.getOperandNumber()] = op.values();
            rewriter.replaceOpWithNewOp<AddIOp>(user, operands);
        }
        else if (isa<AvgOp>(user) || isa<EmitOp>(user))
        {
            return failure();
        }
        else if (isa<SumOp>(user))
        {
            if (!dyn_cast<SumOp>(user).indices())
                continue;
            else
                return failure();
        }
        else if (isa<CountOp>(user))
        {
            if (!dyn_cast<CountOp>(user).indices())
                continue;
            else
                return failure();
        }
        else if (isa<MinOp>(user))
        {
            if (!dyn_cast<MinOp>(user).indices())
                continue;
            else
                return failure();
        }
        else if (isa<MaxOp>(user))
        {
            if (!dyn_cast<MaxOp>(user).indices())
                continue;
            else
                return failure();
        }
        for (auto &u : user->getUses())
            uses.push_back(u);
    }

    if (!op->getUses().empty())
    {
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
    }*/

    return failure();
}

void AddOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool AddOp::predicated()
{
    return pred() != Value();
}

void NeqOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool NeqOp::predicated()
{
    return pred() != Value();
}

void MulOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool MulOp::predicated()
{
    return pred() != Value();
}

void ModOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool ModOp::predicated()
{
    return pred() != Value();
}

void LeqOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool LeqOp::predicated()
{
    return pred() != Value();
}

void LeOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool LeOp::predicated()
{
    return pred() != Value();
}

void GeqOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool GeqOp::predicated()
{
    return pred() != Value();
}

void GeOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool GeOp::predicated()
{
    return pred() != Value();
}

void EqOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool EqOp::predicated()
{
    return pred() != Value();
}


void AndOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool AndOp::predicated()
{
    return pred() != Value();
}

void SubOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool SubOp::predicated()
{
    return pred() != Value();
}

void OrOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool OrOp::predicated()
{
    return pred() != Value();
}

void NotOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool NotOp::predicated()
{
    return pred() != Value();
}

void ScatterOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool ScatterOp::predicated()
{
    return pred() != Value();
}

void ReadOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool ReadOp::predicated()
{
    return pred() != Value();
}

void MinOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool MinOp::predicated()
{
    return pred() != Value();
}

void MaxOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool MaxOp::predicated()
{
    return pred() != Value();
}

void LookupOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool LookupOp::predicated()
{
    return pred() != Value();
}

void InsertOp::predicate(Value pred)
{
    if (predicated() && this->pred() != pred)
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool InsertOp::predicated()
{
    return pred() != Value();
}

void GatherOp::predicate(Value pred)
{
    if (predicated())
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool GatherOp::predicated()
{
    return pred() != Value();
}

void CountOp::predicate(Value pred)
{
    if (predicated())
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool CountOp::predicated()
{
    return pred() != Value();
}

void AvgOp::predicate(Value pred)
{
    if (predicated())
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool AvgOp::predicated()
{
    return pred() != Value();
}

void WriteOp::predicate(Value pred)
{
    if (predicated())
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool WriteOp::predicated()
{
    return pred() != Value();
}

void SumOp::predicate(Value pred)
{
    if (predicated())
        throw ::voila::NotImplementedException(); // TODO
    predMutable().assign(pred);
}

bool SumOp::predicated()
{
    return pred() != Value();
}

void SumOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Value input,
                  Value indices,
                  Value pred)
{
    // TODO: correct return type (tensor if indices)
    SumOp::build(odsBuilder, odsState, odsBuilder.getF64Type(), input, indices, pred);
}

void SumOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Type resType,
                  Value input,
                  Value indices,
                  Value pred)
{
    llvm::SmallVector<int32_t, 3> sizes(1, 1);
    odsState.addOperands(input);
    if (indices)
    {
        odsState.addOperands(indices);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resType);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     Value hashValues,
                     ValueRange values,
                     Value pred)
{
    // TODO: correct return shape
    InsertOp::build(odsBuilder, odsState, values.getTypes(), hashValues, values, pred);
}
void InsertOp::build(mlir::OpBuilder &odsBuilder,
                     mlir::OperationState &odsState,
                     mlir::TypeRange resTypes,
                     mlir::Value hashValues,
                     mlir::ValueRange values,
                     mlir::Value pred)
{
    llvm::SmallVector<int32_t, 4> sizes(1, 1);
    odsState.addOperands(hashValues);
    assert(!values.empty());
    odsState.addOperands(values);
    sizes.push_back(values.size());
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resTypes);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}

void LookupOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     ValueRange values,
                     ValueRange hashtables,
                     Value hashes,
                     Value pred)
{
    LookupOp::build(
        odsBuilder, odsState,
        RankedTensorType::get(hashes.getType().dyn_cast<TensorType>().getShape(), odsBuilder.getIndexType()), values,
        hashtables, hashes, pred);
}
void LookupOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     TypeRange resTypes,
                     ValueRange values,
                     ValueRange hashtables,
                     Value hashes,
                     Value pred)
{
    llvm::SmallVector<int32_t, 5> sizes;
    assert(!values.empty());
    assert(!hashtables.empty());
    assert(hashtables.size() == values.size());
    odsState.addOperands(values);
    sizes.push_back(values.size());
    odsState.addOperands(hashtables);
    sizes.push_back(hashtables.size());
    odsState.addOperands(hashes);
    sizes.push_back(1);
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resTypes);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}

void MinOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Value input,
                  Value indices,
                  Value pred)
{
    // TODO: correct return type (tensor if indices)
    MinOp::build(odsBuilder, odsState, odsBuilder.getF64Type(), input, indices, pred);
}

void MinOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Type resType,
                  Value input,
                  Value indices,
                  Value pred)
{
    llvm::SmallVector<int32_t, 3> sizes(1, 1);
    odsState.addOperands(input);
    if (indices)
    {
        odsState.addOperands(indices);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resType);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}

void MaxOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Value input,
                  Value indices,
                  Value pred)
{
    // TODO: correct return type (tensor if indices)
    MaxOp::build(odsBuilder, odsState, odsBuilder.getF64Type(), input, indices, pred);
}

void MaxOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Type resType,
                  Value input,
                  Value indices,
                  Value pred)
{
    llvm::SmallVector<int32_t, 3> sizes(1, 1);
    odsState.addOperands(input);
    if (indices)
    {
        odsState.addOperands(indices);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resType);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}

void CountOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Value input,
                  Value indices,
                  Value pred)
{
    // TODO: correct return type (tensor if indices)
    CountOp::build(odsBuilder, odsState, odsBuilder.getF64Type(), input, indices, pred);
}

void CountOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  Type resType,
                  Value input,
                  Value indices,
                  Value pred)
{
    llvm::SmallVector<int32_t, 3> sizes(1, 1);
    odsState.addOperands(input);
    if (indices)
    {
        odsState.addOperands(indices);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resType);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}


void AvgOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    Value input,
                    Value indices,
                    Value pred)
{
    // TODO: correct return type (tensor if indices)
    AvgOp::build(odsBuilder, odsState, odsBuilder.getF64Type(), input, indices, pred);
}

void AvgOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    Type resType,
                    Value input,
                    Value indices,
                    Value pred)
{
    llvm::SmallVector<int32_t, 3> sizes(1, 1);
    odsState.addOperands(input);
    if (indices)
    {
        odsState.addOperands(indices);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(resType);
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}