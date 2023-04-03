#pragma once
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

inline bool isInteger(::mlir::Type tpe) { return tpe.isa<::mlir::IntegerType>(); }
inline bool isInteger(::mlir::Value val) { return isInteger(val.getType()); }

inline bool isFloat(::mlir::Type tpe) { return tpe.isa<::mlir::FloatType>(); }
inline bool isFloat(::mlir::Value val) { return isFloat(val.getType()); }

inline bool isTensor(::mlir::Type tpe) { return tpe.isa<::mlir::TensorType>(); }
inline bool isTensor(::mlir::Value val) { return isTensor(val.getType()); }

inline bool isMemRef(::mlir::Type tpe) { return tpe.isa<::mlir::MemRefType>(); }
inline bool isMemRef(::mlir::Value val) { return isMemRef(val.getType()); }

inline bool isIndex(::mlir::Type tpe) { return tpe.isa<::mlir::IndexType>(); }
inline bool isIndex(::mlir::Value val) { return isIndex(val.getType()); }

inline ::mlir::ShapedType asShapedType(::mlir::Type tp) { return tp.dyn_cast<::mlir::ShapedType>(); }
inline ::mlir::ShapedType asShapedType(::mlir::Value val) { return asShapedType(val.getType()); }

inline ::mlir::TensorType asTensorType(::mlir::Type tp) { return tp.dyn_cast<::mlir::TensorType>(); }
inline ::mlir::TensorType asTensorType(::mlir::Value val) { return asTensorType(val.getType()); }

inline bool isShapedType(::mlir::Type tpe) { return tpe.isa<::mlir::ShapedType>(); }

inline bool isShapedType(::mlir::Value val) { return isShapedType(val.getType()); }
inline bool hasStaticShape(::mlir::Type val)
{
    return isShapedType(val) && val.dyn_cast<::mlir::ShapedType>().hasStaticShape();
}
inline bool hasStaticShape(::mlir::Value val) { return hasStaticShape(val.getType()); }

inline llvm::ArrayRef<int64_t> getShape(::mlir::Type tpe) { return tpe.dyn_cast<::mlir::ShapedType>().getShape(); }

inline llvm::ArrayRef<int64_t> getShape(::mlir::Value val) { return getShape(val.getType()); }

inline ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static ::mlir::Value castTensor(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value val, ::mlir::Value res)
{
    ::mlir::Value other;
    if (hasStaticShape(res))
    {
        other = builder.create<::mlir::tensor::EmptyOp>(loc, getShape(res), val.getType());
    }
    else
    {
        ::mlir::SmallVector<::mlir::Value, 1> size;
        size.push_back(builder.create<::mlir::tensor::DimOp>(loc, res, 0));
        other = builder.create<::mlir::tensor::EmptyOp>(loc, ::mlir::ShapedType::kDynamic, val.getType(), size);
    }
    return builder.create<::mlir::linalg::FillOp>(loc, val, other).value();
}

inline std::pair<::mlir::Value, ::mlir::Value>
castShapes(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs)
{
    if (isTensor(lhs) && !isTensor(rhs))
    {
        return {lhs, castTensor(builder, loc, rhs, lhs)};
    }
    else if (isTensor(rhs) && !isTensor(lhs))
    {
        return {castTensor(builder, loc, lhs, rhs), rhs};
    }
    else // no tensors or all tensors as params
    {
        return {lhs, rhs};
    }
}

inline std::pair<::mlir::Value, ::mlir::Value>
castTypes(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs)
{
    auto lhsType = ::mlir::getElementTypeOrSelf(lhs);
    auto rhsType = ::mlir::getElementTypeOrSelf(rhs);

    if (isFloat(lhsType) && !isFloat(rhsType))
    {
        auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, lhsType, rhs);
        return {lhs, castedFlt};
    }
    else if (isFloat(rhsType) && !isFloat(lhsType))
    {
        auto castedFlt = builder.template create<::mlir::arith::FPToSIOp>(loc, rhsType, lhs);
        return {castedFlt, rhs};
    }
    else
    {
        return {lhs, rhs};
    }
}

inline std::pair<::mlir::Value, ::mlir::Value>
canonicalizeValues(::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value lhs, ::mlir::Value rhs)
{
    auto [newLhs, newRhs] = castShapes(builder, loc, lhs, rhs);
    return castTypes(builder, loc, newLhs, newRhs);
}