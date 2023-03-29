#pragma once
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

inline bool hasStaticShape(::mlir::Value val)
{
    return isShapedType(val) && val.getType().dyn_cast<::mlir::ShapedType>().hasStaticShape();
}

inline llvm::ArrayRef<int64_t> getShape(::mlir::Type tpe) { return tpe.dyn_cast<::mlir::ShapedType>().getShape(); }

inline llvm::ArrayRef<int64_t> getShape(::mlir::Value val) { return getShape(val.getType()); }

inline ::mlir::MemRefType convertTensorToMemRef(::mlir::TensorType type)
{
    assert(type.hasRank() && "expected only ranked shapes");
    return ::mlir::MemRefType::get(type.getShape(), type.getElementType());
}