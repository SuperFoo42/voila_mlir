#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"

#include "llvm/ADT/ScopedHashTable.h"      // for ScopedHashTable
#include "llvm/ADT/StringMap.h"            // for StringMap
#include "llvm/ADT/StringRef.h"            // for StringRef, DenseMapInfo
#include "mlir/Dialect/Func/IR/FuncOps.h"  // for FuncOp
#include "mlir/IR/Builders.h"              // for OpBuilder
#include "mlir/IR/BuiltinOps.h"            // for ModuleOp
#include "mlir/IR/OwningOpRef.h"           // for OwningOpRef
#include "mlir/IR/Value.h"                 // for Value

#pragma GCC diagnostic pop

namespace mlir
{
    class MLIRContext;
}

namespace voila
{
    class Program;

    class MLIRGenerator
    {
        ::mlir::OpBuilder builder;
        ::mlir::ModuleOp module;
        llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> symbolTable;
        llvm::StringMap<::mlir::func::FuncOp> funcTable;

        explicit MLIRGenerator(::mlir::MLIRContext &ctx);

        ::mlir::OwningOpRef<::mlir::ModuleOp> generate(const Program &program);

      public:
        static ::mlir::OwningOpRef<::mlir::ModuleOp> mlirGen(::mlir::MLIRContext &ctx, const Program &program);
    };

} // namespace voila
