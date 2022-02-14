#include "mlir/IR/VoilaOps.h"

#include "NotImplementedException.hpp"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

void HashOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value HashOp::predicated()
{
    return pred();
}

void ScatterOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value ScatterOp::predicated()
{
    return pred();
}

void ReadOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value ReadOp::predicated()
{
    return pred();
}

void MinOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value MinOp::predicated()
{
    return pred();
}

void MaxOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value MaxOp::predicated()
{
    return pred();
}

void LookupOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value LookupOp::predicated()
{
    return pred();
}

void InsertOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value InsertOp::predicated()
{
    return pred();
}

void GatherOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value GatherOp::predicated()
{
    return pred();
}

void CountOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value CountOp::predicated()
{
    return pred();
}

void AvgOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value AvgOp::predicated()
{
    return pred();
}

void WriteOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value WriteOp::predicated()
{
    return pred();
}

void SumOp::predicate(Value pred)
{
    predMutable().assign(pred);
}

Value SumOp::predicated()
{
    return pred();
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
void HashOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, Value pred)
{
    llvm::SmallVector<int32_t, 2> sizes(1, inputs.size());
    odsState.addOperands(inputs);

    if (pred)
    {
        odsState.addOperands(pred);
        sizes.push_back(1);
    }
    else
    {
        sizes.push_back(0);
    }
    odsState.addTypes(odsBuilder.getI64Type());
    odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr(sizes));
}