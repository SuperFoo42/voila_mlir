#pragma once

#include "ASTNodes.hpp"
#include "Program.hpp"

#include <variant>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/VoilaOps.h"
#pragma GCC diagnostic pop
namespace voila
{
    using namespace ast;

    class MLIRGenerator
    {
        ::mlir::OpBuilder builder;
        ::mlir::ModuleOp module;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        std::unordered_map<std::string, ::mlir::FuncOp> funcTable;

        explicit MLIRGenerator(::mlir::MLIRContext &ctx);

        ::mlir::OwningModuleRef generate(const Program &program);

      public:
        static ::mlir::OwningModuleRef mlirGen(::mlir::MLIRContext &ctx, const Program &program);
    };

} // namespace voila
