#pragma once
#include "VariableAlreadyDeclaredException.hpp" // for VariableAlreadyDecla...
#include "ast/ASTNode.hpp"                      // for ASTNode (ptr only)
#include "ast/ASTVisitor.hpp"                   // for ASTVisitor
#include "ast/Comparison.hpp"                   // for Comparison
#include <cstdint>                              // for int64_t
#include <memory>                               // for shared_ptr
#include <mlir/IR/Builders.h>                   // for OpBuilder
#include <variant>                              // for get, variant, monostate
#include <vector>                               // for vector

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"       // for IntegerType, TensorType
#include "mlir/IR/BuiltinTypes.h"     // for IntegerType, TensorType
#include "mlir/IR/Location.h"         // for Location
#include "mlir/IR/Types.h"            // for Type
#include "mlir/IR/Value.h"            // for Value
#include "llvm/ADT/ArrayRef.h"        // for ArrayRef
#include "llvm/ADT/SmallVector.h"     // for SmallVector
#include "llvm/ADT/StringMap.h"       // for StringMap
#include "llvm/ADT/StringRef.h"       // for StringRef, DenseMapInfo
#include <llvm/ADT/ScopedHashTable.h> // for ScopedHashTable

#pragma GCC diagnostic pop

namespace mlir
{
    struct LogicalResult;
}

namespace voila
{
    class Type;
    class TypeInferer;
    namespace ast
    {
        class Add;
        class AggrAvg;
        class AggrCnt;
        class AggrMax;
        class AggrMin;
        class AggrSum;
        class And;
        class Assign;
        class BooleanConst;
        class Div;
        class Emit;
        class Eq;
        class Expression;
        class FltConst;
        class Fun;
        class FunctionCall;
        class Gather;
        class Ge;
        class Geq;
        class Hash;
        class Insert;
        class IntConst;
        class Le;
        class Leq;
        class Lookup;
        class Loop;
        class Main;
        class Mod;
        class Mul;
        class Neq;
        class Not;
        class Or;
        class Predicate;
        class Read;
        class Ref;
        class Scatter;
        class Selection;
        class Statement;
        class StatementWrapper;
        class StrConst;
        class Sub;
        class TupleCreate;
        class TupleGet;
        class Variable;
        class Write;
    } // namespace ast
} // namespace voila

namespace voila::mlir
{
    class MLIRGeneratorImpl : public ast::ASTVisitor
    {
        ::mlir::OpBuilder &builder;
        ::mlir::ModuleOp &module;
        llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable;
        llvm::StringMap<::mlir::func::FuncOp> &funcTable;
        const TypeInferer &inferer;
        using result_variant =
            std::variant<std::monostate,
                         ::mlir::ModuleOp,
                         ::mlir::Value,
                         ::mlir::SmallVector<::mlir::Value>, // FIXME: ugly, but needed because ValueRange does not
                                                             // extend lifetime of underlying object
                         ::mlir::Type,
                         ::mlir::LogicalResult,
                         ::mlir::func::FuncOp>;
        result_variant result;

        // helper functions to map ast types to mlir
        ::mlir::Location loc(ast::Location loc);

        std::vector<::mlir::Type> getTypes(const ast::ASTNode &node);

        ::mlir::Type convert(const Type &t);

        // TODO: is this correct?
        void declare(llvm::StringRef var, ::mlir::Value value)
        {
            (void)module;
            if (symbolTable.count(var))
                throw ast::VariableAlreadyDeclaredException();
            symbolTable.insert(var, value);
        }

        void mlirGenBody(const std::vector<ast::Statement> &block);

        result_variant visitor_gen(const ast::Statement &node);

        result_variant visitor_gen(const ast::Expression &node);

        result_variant visitor_gen(const std::vector<ast::Expression> &nodes);

        result_variant visitor_gen(const std::vector<ast::Statement> &nodes);

        static llvm::ArrayRef<int64_t> getShape(const ::mlir::Value &lhs, const ::mlir::Value &rhs);

        template <class Op>::mlir::Value getCmpOp(const ast::Comparison &cmpNode)
        {
            auto location = loc(cmpNode.get_location());
            auto lhs = std::get<::mlir::Value>(visitor_gen(cmpNode.lhs()));
            auto rhs = std::get<::mlir::Value>(visitor_gen(cmpNode.rhs()));
            if (lhs.getType().isa<::mlir::TensorType>() || rhs.getType().isa<::mlir::TensorType>())
            {
                ::mlir::ArrayRef<int64_t> shape;
                shape = getShape(lhs, rhs);

                return builder.create<Op>(location, ::mlir::RankedTensorType::get(shape, builder.getI1Type()), lhs,
                                          rhs);
            }
            else
                return builder.create<Op>(location, builder.getI1Type(), lhs, rhs);
        }

        ::mlir::Type getScalarType(const ast::ASTNode &node);

        ::mlir::Type scalarConvert(const std::shared_ptr<::voila::Type> &t);
        ::mlir::Type scalarConvert(const ::voila::Type &t);

      public:
        MLIRGeneratorImpl(::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp &module,
                          ::llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                          llvm::StringMap<::mlir::func::FuncOp> &funcTable,
                          const TypeInferer &inferer);

        result_variant getValue() { return result; }

        void operator()(const ast::AggrSum &sum) final;
        void operator()(const ast::AggrCnt &cnt) final;
        void operator()(const ast::AggrMin &min) final;
        void operator()(const ast::AggrMax &max) final;
        void operator()(const ast::AggrAvg &avg) final;
        void operator()(const ast::Write &write) final;
        void operator()(const ast::Scatter &scatter) final;

        void operator()(const ast::Assign &assign) final;
        void operator()(const ast::Emit &emit) final;
        void operator()(const ast::Loop &loop) final;
        void operator()(const ast::StatementWrapper &wrapper) final;
        void operator()(const ast::Add &add) final;
        void operator()(const ast::Sub &sub) final;
        void operator()(const ast::Mul &mul) final;
        void operator()(const ast::Div &div) final;
        void operator()(const ast::Mod &mod) final;
        void operator()(const ast::Eq &eq) final;
        void operator()(const ast::Neq &neq) final;
        void operator()(const ast::Le &le) final;
        void operator()(const ast::Ge &ge) final;
        void operator()(const ast::Leq &leq) final;
        void operator()(const ast::Geq &geq) final;
        void operator()(const ast::And &anAnd) final;
        void operator()(const ast::Or &anOr) final;
        void operator()(const ast::Not &aNot) final;
        void operator()(const ast::IntConst &intConst) final;
        void operator()(const ast::BooleanConst &booleanConst) final;
        void operator()(const ast::FltConst &fltConst) final;
        void operator()(const ast::StrConst &aConst) final;
        void operator()(const ast::Read &read) final;
        void operator()(const ast::Gather &gather) final;
        void operator()(const ast::Ref &param) final;
        void operator()(const ast::TupleGet &get) final;
        void operator()(const ast::TupleCreate &create) final;
        void operator()(const ast::Fun &fun) final;
        void operator()(const ast::Main &main) final;
        void operator()(const ast::Selection &selection) final;
        void operator()(const ast::Variable &variable) final;
        void operator()(const ast::Predicate &pred) final;
        void operator()(const ast::Hash &hash) final;
        void operator()(const ast::Lookup &lookup) override;
        void operator()(const ast::Insert &insert) override;
        void operator()(const ast::FunctionCall &call) final;
        void operator()(ast::FunctionCall &call) override;
    };
} // namespace voila::mlir