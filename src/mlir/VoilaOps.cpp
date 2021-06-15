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

void mlir::voila::AddOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::SubOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::MulOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::DivOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::ModOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::AndOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::OrOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::NotOp::inferShapes()
{
    getResult().setType(value().getType());
}

void mlir::voila::EqOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::NeqOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::LeOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::LeqOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::GeOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}

void mlir::voila::GeqOp::inferShapes()
{
    getResult().setType(::inferShapesFromBinaryOp(lhs().getType(), rhs().getType()));
}