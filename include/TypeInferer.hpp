#pragma once
#include "Types.hpp"          // for DataType (ptr only), type_id_t, Type (...
#include <ast/ASTNode.hpp>    // for ASTVisitor
#include <ast/ASTVisitor.hpp> // for ASTVisitor
#include <cstddef>            // for size_t
#include <memory>             // for shared_ptr
#include <unordered_map>      // for unordered_map
#include <utility>            // for pair
#include <vector>             // for vector

namespace voila
{
    class Program;

    namespace ast
    {
        class Add;
        class AggrAvg;
        class AggrCnt;
        class AggrMax;
        class AggrMin;
        class AggrSum;
        class And;
        class Arithmetic;
        class Assign;
        class BooleanConst;
        class Comparison;
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

    // TODO pimpl
    class TypeInferer : public ast::ASTVisitor<>
    {
      public:
        explicit TypeInferer(Program *prog);

        void operator()(std::shared_ptr<ast::Write> write) final;
        void operator()(std::shared_ptr<ast::Scatter> scatter) final;
        void operator()(std::shared_ptr<ast::FunctionCall> call) final;
        void operator()(std::shared_ptr<ast::Assign> assign) final;
        void operator()(std::shared_ptr<ast::Emit> emit) final;
        void operator()(std::shared_ptr<ast::Loop> loop) final;
        void operator()(std::shared_ptr<ast::IntConst> aConst) final;
        void operator()(std::shared_ptr<ast::BooleanConst> aConst) final;
        void operator()(std::shared_ptr<ast::FltConst> aConst) final;
        void operator()(std::shared_ptr<ast::StrConst> aConst) final;
        void operator()(std::shared_ptr<ast::Read> read) final;
        void operator()(std::shared_ptr<ast::Gather> gather) final;
        void operator()(std::shared_ptr<ast::Ref> param) final;
        void operator()(std::shared_ptr<ast::Fun> fun) final;
        void operator()(std::shared_ptr<ast::Main> main) final;
        void operator()(std::shared_ptr<ast::Selection> selection) final;
        void operator()(std::shared_ptr<ast::Variable> var) final;

        void operator()(std::shared_ptr<ast::Add> var) final;
        void operator()(std::shared_ptr<ast::Sub> sub) final;
        void operator()(std::shared_ptr<ast::Mul> mul) final;
        void operator()(std::shared_ptr<ast::Div> div1) final;
        void operator()(std::shared_ptr<ast::Mod> mod) final;
        void operator()(std::shared_ptr<ast::AggrSum> sum) final;
        void operator()(std::shared_ptr<ast::AggrCnt> cnt) final;
        void operator()(std::shared_ptr<ast::AggrMin> aggrMin) final;
        void operator()(std::shared_ptr<ast::AggrMax> aggrMax) final;
        void operator()(std::shared_ptr<ast::AggrAvg> avg) final;
        void operator()(std::shared_ptr<ast::Eq> eq) final;
        void operator()(std::shared_ptr<ast::Neq> neq) final;
        void operator()(std::shared_ptr<ast::Le> le) final;
        void operator()(std::shared_ptr<ast::Ge> ge) final;
        void operator()(std::shared_ptr<ast::Leq> leq) final;
        void operator()(std::shared_ptr<ast::Geq> geq) final;
        void operator()(std::shared_ptr<ast::And> anAnd) final;
        void operator()(std::shared_ptr<ast::Or> anOr) final;
        void operator()(std::shared_ptr<ast::Not> aNot) final;
        void operator()(std::shared_ptr<ast::Predicate> pred) final;
        void operator()(std::shared_ptr<ast::StatementWrapper> wrapper) final;
        void operator()(std::shared_ptr<ast::Hash> hash) final;

        std::shared_ptr<Type> get_type(const ast::ASTNodeVariant &node) const;

        void set_arity(const ast::ASTNodeVariant &node, size_t ar);
        void set_type(const ast::ASTNodeVariant &node, DataType type);

        void insertNewType(const ast::ASTNodeVariant &node, DataType t, Arity ar);
        void insertNewTypeAs(const ast::ASTNodeVariant &node, const Type &t);
        void operator()(std::shared_ptr<ast::Lookup> lookup) override;
        void operator()(std::shared_ptr<ast::Insert> insert) override;

        using ast::ASTVisitor<void>::operator();

        std::vector<std::shared_ptr<Type>> types;
        Program *prog;

      private:
        static DataType convert(DataType, DataType);

        static bool compatible(DataType, DataType);

        /**
         * Check whether first type is convertible to second type
         * @param t1 Type to convert
         * @param t2 destination type
         * @return
         */
        static bool convertible(DataType t1, DataType t2);

        void insertNewFuncType(const ast::ASTNodeVariant &node,
                               std::vector<size_t> typeParamIDs,
                               DataType returnT,
                               Arity returnAr);

        type_id_t get_type_id(const ast::ASTNodeVariant &node);

        void
        insertNewFuncType(const ast::ASTNodeVariant &node, std::vector<type_id_t> typeParamIDs, type_id_t returnTypeID);

        void insertNewFuncType(const ast::ASTNodeVariant &node,
                               std::vector<type_id_t> typeParamIDs,
                               const std::vector<std::pair<DataType, Arity>> &returnTypes);

        void insertNewFuncType(const ast::ASTNodeVariant &node,
                               std::vector<type_id_t> typeParamIDs,
                               std::vector<type_id_t> returnTypeIDs);

        std::unordered_map<ast::ASTNodeVariant, type_id_t> typeIDs;

        void unify(const ast::ASTNodeVariant &t1, const ast::ASTNodeVariant &t2);
        void unify(const std::vector<ast::ASTNodeVariant> &t1, const ast::ASTNodeVariant &t2);

        template <class T> void visitArithmetic(T arithmetic);

        template <class T> void visitComparison(T comparison);
    };
} // namespace voila
