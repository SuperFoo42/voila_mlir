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
#pragma GCC diagnostic pop
namespace voila
{
    class Program;

    class MLIRGenerator
    {
        ::mlir::OpBuilder builder;
        ::mlir::ModuleOp module;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        llvm::StringMap<::mlir::FuncOp> funcTable;

        explicit MLIRGenerator(::mlir::MLIRContext &ctx);

        ::mlir::OwningModuleRef generate(const Program &program);

      public:
        static ::mlir::OwningModuleRef mlirGen(::mlir::MLIRContext &ctx, const Program &program);
    };

} // namespace voila
