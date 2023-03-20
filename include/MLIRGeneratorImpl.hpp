#pragma once
#include "VariableAlreadyDeclaredException.hpp" // for VariableAlreadyDecla...
#include "ast/ASTNode.hpp"                      // for ASTNode (ptr only)
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include <cstdint>            // for int64_t
#include <memory>             // for shared_ptr
#include <mlir/IR/Builders.h> // for OpBuilder
#include <variant>            // for get, variant, monostate
#include <vector>             // for vector

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "../src/MlirGenerationException.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/IR/BuiltinOps.h"   // for IntegerType, TensorType
#include "mlir/IR/BuiltinTypes.h" // for IntegerType, TensorType
#include "mlir/IR/Location.h"     // for Location
#include "mlir/IR/TypeUtilities.h"
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

using result_variant =
    std::variant<std::monostate,
                 ::mlir::ModuleOp,
                 ::mlir::Value,
                 ::mlir::ValueRange, // does not extend lifetime of underlying object //TODO: all types contained?
                 //::mlir::SmallVector<::mlir::Value>,
                 ::mlir::Type,
                 ::mlir::LogicalResult,
                 ::mlir::func::FuncOp>;

namespace
{
    template <typename T> struct aggr_ret_type
    {
        ::mlir::Type operator()(::mlir::OpBuilder &builder, ::mlir::Value &expr)
        {
            if (::mlir::getElementTypeOrSelf(expr).isIntOrIndex())
                return builder.getI64Type();
            else if (::mlir::getElementTypeOrSelf(expr).isIntOrFloat())
                return builder.getF64Type();
            else
                throw voila::ast::MLIRGenerationException();
        }
    };

    template <> struct aggr_ret_type<voila::ast::AggrCnt>
    {
        ::mlir::Type operator()(::mlir::OpBuilder &builder, ::mlir::Value &) { return builder.getI64Type(); }
    };

    template <> struct aggr_ret_type<voila::ast::AggrAvg>
    {
        ::mlir::Type operator()(::mlir::OpBuilder &builder, ::mlir::Value &) { return builder.getF64Type(); }
    };

    template <typename T> struct aggr_to_op;
    template <> struct aggr_to_op<voila::ast::AggrSum>
    {
        using type = ::mlir::voila::SumOp;
    };
    template <> struct aggr_to_op<voila::ast::AggrAvg>
    {
        using type = ::mlir::voila::AvgOp;
    };
    template <> struct aggr_to_op<voila::ast::AggrMin>
    {
        using type = ::mlir::voila::MinOp;
    };
    template <> struct aggr_to_op<voila::ast::AggrMax>
    {
        using type = ::mlir::voila::MaxOp;
    };
    template <> struct aggr_to_op<voila::ast::AggrCnt>
    {
        using type = ::mlir::voila::CountOp;
    };


    template <typename T> using aggr_to_op_t = typename aggr_to_op<T>::type;

    template <typename T> struct cmp_to_op;
    template <> struct cmp_to_op<voila::ast::Eq>
    {
        using type = ::mlir::voila::EqOp;
    };
    template <> struct cmp_to_op<voila::ast::Neq>
    {
        using type = ::mlir::voila::NeqOp;
    };
    template <> struct cmp_to_op<voila::ast::Ge>
    {
        using type = ::mlir::voila::GeOp;
    };
    template <> struct cmp_to_op<voila::ast::Geq>
    {
        using type = ::mlir::voila::GeqOp;
    };
    template <> struct cmp_to_op<voila::ast::Le>
    {
        using type = ::mlir::voila::LeOp;
    };
    template <> struct cmp_to_op<voila::ast::Leq>
    {
        using type = ::mlir::voila::LeqOp;
    };

    template <typename T> using cmp_to_op_t = typename cmp_to_op<T>::type;

} // namespace

namespace voila::mlir
{
    class MLIRGeneratorImpl : public ast::ASTVisitor<result_variant>
    {
        ::mlir::OpBuilder &builder;
        ::mlir::ModuleOp &module;
        llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable;
        llvm::StringMap<::mlir::func::FuncOp> &funcTable;
        const TypeInferer &inferer;

        // helper functions to map ast types to mlir
        ::mlir::Location loc(ast::Location loc);

        std::vector<::mlir::Type> getTypes(const ast::ASTNodeVariant &node);

        ::mlir::Type convert(const Type &t);

        // TODO: is this correct?
        void declare(llvm::StringRef var, ::mlir::Value value)
        {
            (void)module;
            if (symbolTable.count(var))
                throw ast::VariableAlreadyDeclaredException();
            symbolTable.insert(var, value);
        }

        void mlirGenBody(const std::vector<ast::ASTNodeVariant> &block);

        static llvm::ArrayRef<int64_t> getShape(const ::mlir::Value &lhs, const ::mlir::Value &rhs);

        template <class Cmp>::mlir::Value getCmpOp(const Cmp &cmpNode)
        {
            auto location = loc(cmpNode.get_location());
            auto lhs = std::get<::mlir::Value>(std::visit(*this, cmpNode.lhs()));
            auto rhs = std::get<::mlir::Value>(std::visit(*this, cmpNode.rhs()));
            ::mlir::Type retType =
                (lhs.getType().template isa<::mlir::TensorType>() || rhs.getType().template isa<::mlir::TensorType>())
                    ?
                    static_cast<::mlir::Type>(::mlir::RankedTensorType::get(getShape(lhs, rhs), builder.getI1Type()))
                    : static_cast<::mlir::Type>(builder.getI1Type());

            return builder.create<cmp_to_op_t<Cmp>>(location, retType, lhs, rhs);
        }

        ::mlir::Type getScalarType(const ast::ASTNodeVariant &node);

        ::mlir::Type scalarConvert(const std::shared_ptr<::voila::Type> &t);
        ::mlir::Type scalarConvert(const ::voila::Type &t);

        template <class AggrType>
        [[nodiscard]] ::mlir::Type getResultType(std::shared_ptr<AggrType> &aggr, ::mlir::Value &expr)
        {
            auto type = aggr_ret_type<AggrType>()(builder, expr);
            return aggr->groups()
                       ? static_cast<::mlir::Type>(type)
                       : static_cast<::mlir::Type>(::mlir::RankedTensorType::get(::mlir::ShapedType::kDynamic, type));
        }

        template <class T>::mlir::Value createAggr(std::shared_ptr<T> &aggr)
        {
            auto location = loc(aggr->get_location());

            ::mlir::Value expr = std::get<::mlir::Value>(std::visit(*this, aggr->src()));

            auto idxs = std::get<::mlir::Value>(std::visit(*this, aggr->groups()));

            return builder.create<aggr_to_op_t<T>>(location, getResultType(aggr, expr), expr, idxs);
        }

      public:
        MLIRGeneratorImpl(::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp &module,
                          ::llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                          llvm::StringMap<::mlir::func::FuncOp> &funcTable,
                          const TypeInferer &inferer);

        result_variant operator()(std::shared_ptr<ast::AggrSum> sum) final;
        result_variant operator()(std::shared_ptr<ast::AggrCnt> cnt) final;
        result_variant operator()(std::shared_ptr<ast::AggrMin> min) final;
        result_variant operator()(std::shared_ptr<ast::AggrMax> max) final;
        result_variant operator()(std::shared_ptr<ast::AggrAvg> avg) final;
        result_variant operator()(std::shared_ptr<ast::Write> write) final;
        result_variant operator()(std::shared_ptr<ast::Scatter> scatter) final;

        result_variant operator()(std::shared_ptr<ast::Assign> assign) final;
        result_variant operator()(std::shared_ptr<ast::Emit> emit) final;
        result_variant operator()(std::shared_ptr<ast::Loop> loop) final;
        result_variant operator()(std::shared_ptr<ast::StatementWrapper> wrapper) final;
        result_variant operator()(std::shared_ptr<ast::Add> add) final;
        result_variant operator()(std::shared_ptr<ast::Sub> sub) final;
        result_variant operator()(std::shared_ptr<ast::Mul> mul) final;
        result_variant operator()(std::shared_ptr<ast::Div> div) final;
        result_variant operator()(std::shared_ptr<ast::Mod> mod) final;
        result_variant operator()(std::shared_ptr<ast::Eq> eq) final;
        result_variant operator()(std::shared_ptr<ast::Neq> neq) final;
        result_variant operator()(std::shared_ptr<ast::Le> le) final;
        result_variant operator()(std::shared_ptr<ast::Ge> ge) final;
        result_variant operator()(std::shared_ptr<ast::Leq> leq) final;
        result_variant operator()(std::shared_ptr<ast::Geq> geq) final;
        result_variant operator()(std::shared_ptr<ast::And> anAnd) final;
        result_variant operator()(std::shared_ptr<ast::Or> anOr) final;
        result_variant operator()(std::shared_ptr<ast::Not> aNot) final;
        result_variant operator()(std::shared_ptr<ast::IntConst> intConst) final;
        result_variant operator()(std::shared_ptr<ast::BooleanConst> booleanConst) final;
        result_variant operator()(std::shared_ptr<ast::FltConst> fltConst) final;
        result_variant operator()(std::shared_ptr<ast::StrConst> aConst) final;
        result_variant operator()(std::shared_ptr<ast::Read> read) final;
        result_variant operator()(std::shared_ptr<ast::Gather> gather) final;
        result_variant operator()(std::shared_ptr<ast::Ref> param) final;
        result_variant operator()(std::shared_ptr<ast::Fun> fun) final;
        result_variant operator()(std::shared_ptr<ast::Main> main) final;
        result_variant operator()(std::shared_ptr<ast::Selection> selection) final;
        result_variant operator()(std::shared_ptr<ast::Variable> variable) final;
        result_variant operator()(std::shared_ptr<ast::Predicate> pred) final;
        result_variant operator()(std::shared_ptr<ast::Hash> hash) final;
        result_variant operator()(std::shared_ptr<ast::Lookup> lookup) final;
        result_variant operator()(std::shared_ptr<ast::Insert> insert) final;
        result_variant operator()(std::shared_ptr<ast::FunctionCall> call) final;

        result_variant operator()(std::monostate) final;
    };
} // namespace voila::mlir