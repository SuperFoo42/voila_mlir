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
#pragma GCC diagnostic pop
namespace voila
{
    using namespace ast;

    class MLIRGenerator
    {
        ::mlir::OpBuilder builder;
        ::mlir::ModuleOp module;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

        explicit MLIRGenerator(::mlir::MLIRContext &ctx) : builder{&ctx}
        {
            (void) this->builder;
        }

        ::mlir::OwningModuleRef generate(const Program &program);

      public:
        static ::mlir::OwningModuleRef mlirGen(::mlir::MLIRContext &ctx, const Program &program)
        {
            MLIRGenerator generator(ctx);
            return generator.generate(program);
        }
    };

    class MLIRGeneratorImpl : public ASTVisitor
    {
        ::mlir::OpBuilder &builder;
        ::mlir::ModuleOp &module;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symbolTable;
        const TypeInferer &inferer;
        using result_variant = std::variant<std::monostate,
                                            ::mlir::ModuleOp,
                                            ::mlir::Value,
                                            ::mlir::Type,
                                            ::mlir::LogicalResult,
                                            ::mlir::FuncOp>;
        result_variant result;

        // helper functions to map ast types to mlir
        ::mlir::Location loc(Location loc)
        {
            return mlir::FileLineColLoc::get(builder.getIdentifier(*loc.begin.filename), loc.begin.line,
                                             loc.begin.column);
        }

        ::mlir::Type getType(const ASTNode &node)
        {
            const auto astType = inferer.get_type(node);
            return convert(astType);
        }

        // TODO: implement
        ::mlir::Type convert(Type t)
        {
            (void) t;
            return nullptr;
        }

        // TODO: is this correct?
        mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value)
        {
            (void)module;
            if (symbolTable.count(var))
                return mlir::failure();
            symbolTable.insert(var, value);
            return mlir::success();
        }

      public:
        MLIRGeneratorImpl(::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp &module,
                          llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symbolTable,
                          const TypeInferer &inferer) :
            builder{builder}, module{module}, symbolTable{symbolTable}, inferer{inferer}, result{}
        {
            (void) module;
        }

        result_variant getValue()
        {
            return result;
        }

        void operator()(const AggrSum &sum) override;
        void operator()(const AggrCnt &cnt) override;
        void operator()(const AggrMin &min) override;
        void operator()(const AggrMax &max) override;
        void operator()(const AggrAvg &avg) override;
        void operator()(const Write &write) override;
        void operator()(const Scatter &scatter) override;
        void operator()(const FunctionCall &call) override;
        void operator()(const Assign &assign) override;
        void operator()(const Emit &emit) override;
        void operator()(const Loop &loop) override;
        void operator()(const StatementWrapper &wrapper) override;
        void operator()(const Add &add) override;
        void operator()(const Sub &sub) override;
        void operator()(const Mul &mul) override;
        void operator()(const Div &div) override;
        void operator()(const Mod &mod) override;
        void operator()(const Eq &eq) override;
        void operator()(const Neq &neq) override;
        void operator()(const Le &le) override;
        void operator()(const Ge &ge) override;
        void operator()(const Leq &leq) override;
        void operator()(const Geq &geq) override;
        void operator()(const And &anAnd) override;
        void operator()(const Or &anOr) override;
        void operator()(const Not &aNot) override;
        void operator()(const IntConst &aConst) override;
        void operator()(const BooleanConst &aConst) override;
        void operator()(const FltConst &aConst) override;
        void operator()(const StrConst &aConst) override;
        void operator()(const Read &read) override;
        void operator()(const Gather &gather) override;
        void operator()(const Ref &param) override;
        void operator()(const TupleGet &get) override;
        void operator()(const TupleCreate &create) override;
        void operator()(const Fun &fun) override;
        void operator()(const Main &main) override;
        void operator()(const Selection &selection) override;
        void operator()(const Variable &variable) override;
    };
} // namespace voila
