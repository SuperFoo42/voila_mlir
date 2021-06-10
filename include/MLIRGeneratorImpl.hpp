#pragma once
#include "ASTNodes.hpp"
#include "Program.hpp"

#include <NotInferedException.hpp>
#include <variant>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "VariableAlreadyDeclaredException.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/VoilaOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#pragma GCC diagnostic pop
namespace voila::mlir
{
    class MLIRGeneratorImpl : public ast::ASTVisitor
    {
        ::mlir::OpBuilder &builder;
        ::mlir::ModuleOp &module;
        llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable;
        std::unordered_map<std::string, ::mlir::FuncOp> &funcTable;
        const TypeInferer &inferer;
        using result_variant = std::variant<std::monostate,
                                            ::mlir::ModuleOp,
                                            ::mlir::Value,
                                            ::mlir::Type,
                                            ::mlir::LogicalResult,
                                            ::mlir::FuncOp>;
        result_variant result;

        // helper functions to map ast types to mlir
        ::mlir::Location loc(ast::Location loc);

        ::mlir::Type getType(const ast::ASTNode &node);

        // TODO: implement
        ::mlir::Type convert(const Type &t);

        // TODO: is this correct?
        void declare(llvm::StringRef var, ::mlir::Value value)
        {
            (void) module;
            if (symbolTable.count(var))
                throw ast::VariableAlreadyDeclaredException();
            symbolTable.insert(var, value);
        }

        void mlirGenBody(const std::vector<ast::Statement> &block);

        result_variant visitor_gen(const ast::Statement &node);

        result_variant visitor_gen(const ast::Expression &node);

      public:
        MLIRGeneratorImpl(::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp &module,
                          ::llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                          std::unordered_map<std::string, ::mlir::FuncOp> &funcTable,
                          const TypeInferer &inferer) :
            builder{builder}, module{module}, symbolTable{symbolTable}, funcTable{funcTable}, inferer{inferer}, result{}
        {
            (void) module;
        }

        result_variant getValue()
        {
            return result;
        }

        void operator()(const ast::AggrSum &sum) override;
        void operator()(const ast::AggrCnt &cnt) override;
        void operator()(const ast::AggrMin &min) override;
        void operator()(const ast::AggrMax &max) override;
        void operator()(const ast::AggrAvg &avg) override;
        void operator()(const ast::Write &write) override;
        void operator()(const ast::Scatter &scatter) override;
        void operator()(const ast::FunctionCall &call) override;
        void operator()(const ast::Assign &assign) override;
        void operator()(const ast::Emit &emit) override;
        void operator()(const ast::Loop &loop) override;
        void operator()(const ast::StatementWrapper &wrapper) override;
        void operator()(const ast::Add &add) override;
        void operator()(const ast::Sub &sub) override;
        void operator()(const ast::Mul &mul) override;
        void operator()(const ast::Div &div) override;
        void operator()(const ast::Mod &mod) override;
        void operator()(const ast::Eq &eq) override;
        void operator()(const ast::Neq &neq) override;
        void operator()(const ast::Le &le) override;
        void operator()(const ast::Ge &ge) override;
        void operator()(const ast::Leq &leq) override;
        void operator()(const ast::Geq &geq) override;
        void operator()(const ast::And &anAnd) override;
        void operator()(const ast::Or &anOr) override;
        void operator()(const ast::Not &aNot) override;
        void operator()(const ast::IntConst &intConst) override;
        void operator()(const ast::BooleanConst &booleanConst) override;
        void operator()(const ast::FltConst &fltConst) override;
        void operator()(const ast::StrConst &aConst) override;
        void operator()(const ast::Read &read) override;
        void operator()(const ast::Gather &gather) override;
        void operator()(const ast::Ref &param) override;
        void operator()(const ast::TupleGet &get) override;
        void operator()(const ast::TupleCreate &create) override;
        void operator()(const ast::Fun &fun) override;
        void operator()(const ast::Main &main) override;
        void operator()(const ast::Selection &selection) override;
        void operator()(const ast::Variable &variable) override;
    };
} // namespace voila::mlir
