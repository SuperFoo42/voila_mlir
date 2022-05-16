#pragma once

#include "ASTNodes.hpp"

#include <variant>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#pragma GCC diagnostic pop

namespace voila {
    class Program;

    class MLIRGenerator {
        ::mlir::OpBuilder builder;
        ::mlir::ModuleOp module;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        llvm::StringMap<::mlir::func::FuncOp> funcTable;

        explicit MLIRGenerator(::mlir::MLIRContext &ctx);

        mlir::OwningOpRef<mlir::ModuleOp> generate(const Program &program);

    public:
        static mlir::OwningOpRef<mlir::ModuleOp> mlirGen(::mlir::MLIRContext &ctx, const Program &program);
    };

} // namespace voila
