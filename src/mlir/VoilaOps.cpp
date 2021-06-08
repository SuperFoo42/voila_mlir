#include "mlir/VoilaOps.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/VoilaDialect.h"

#define GET_OP_CLASSES
#include "mlir/VoilaOps.cpp.inc"

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

void mlir::voila::AddOp::inferShapes()
{
    if (lhs().getType().isF64())
    {
        getResult().setType(lhs().getType());
    }
    else
    {
        getResult().setType(rhs().getType());
    }
}

void mlir::voila::SubOp::inferShapes()
{
    if (lhs().getType().isF64())
    {
        getResult().setType(lhs().getType());
    }
    else
    {
        getResult().setType(rhs().getType());
    }
}

void mlir::voila::MulOp::inferShapes()
{
    if (lhs().getType().isF64())
    {
        getResult().setType(lhs().getType());
    }
    else
    {
        getResult().setType(rhs().getType());
    }
}

void mlir::voila::DivOp::inferShapes()
{
    if (lhs().getType().isF64())
    {
        getResult().setType(lhs().getType());
    }
    else
    {
        getResult().setType(rhs().getType());
    }
}

void mlir::voila::ModOp::inferShapes()
{
    if (lhs().getType().isF64())
    {
        getResult().setType(lhs().getType());
    }
    else
    {
        getResult().setType(rhs().getType());
    }
}

void mlir::voila::AndOp::inferShapes()
{
    //back propagate partially known shapes
/*
    const auto lhs_size = lhs().getType().dyn_cast<RankedTensorType>().getShape().front();
    const auto rhs_size = lhs().getType().dyn_cast<RankedTensorType>().getShape().front();
    if (rhs_size < 0 && lhs_size >= 0)
        rhs().setType(RankedTensorType::get(lhs_size, IntegerType::get(lhs().getType().getContext(), 1)));
    if (lhs_size < 0 && rhs_size >= 0)
        lhs().setType(RankedTensorType::get(rhs_size, IntegerType::get(rhs().getType().getContext(), 1)));
*/

    getResult().setType(lhs().getType());
}

void mlir::voila::OrOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::NotOp::inferShapes()
{
    getResult().setType(value().getType());
}

void mlir::voila::EqOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::NeqOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::LeOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::LeqOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::GeOp::inferShapes()
{
    getResult().setType(lhs().getType());
}

void mlir::voila::GeqOp::inferShapes()
{
    getResult().setType(lhs().getType());
}