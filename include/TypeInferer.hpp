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
        class Assign;
        class BooleanConst;
        class Div;
        class Emit;
        class Eq;
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
        class StatementWrapper;
        class StrConst;
        class Sub;
        class Variable;
        class Write;
    } // namespace ast

    // TODO pimpl
    class TypeInferer : public ast::ASTVisitor<TypeInferer, void>
    {
      public:

        explicit TypeInferer(Program *prog);

        return_type visit_impl(std::shared_ptr<ast::Write> write);
        return_type visit_impl(std::shared_ptr<ast::Scatter> scatter);
        return_type visit_impl(std::shared_ptr<ast::FunctionCall> call);
        return_type visit_impl(std::shared_ptr<ast::Assign> assign);
        return_type visit_impl(std::shared_ptr<ast::Emit> emit);
        return_type visit_impl(std::shared_ptr<ast::Loop> loop);
        return_type visit_impl(std::shared_ptr<ast::IntConst> aConst);
        return_type visit_impl(std::shared_ptr<ast::BooleanConst> aConst);
        return_type visit_impl(std::shared_ptr<ast::FltConst> aConst);
        return_type visit_impl(std::shared_ptr<ast::StrConst> aConst);
        return_type visit_impl(std::shared_ptr<ast::Read> read);
        return_type visit_impl(std::shared_ptr<ast::Gather> gather);
        return_type visit_impl(std::shared_ptr<ast::Ref> param);
        return_type visit_impl(std::shared_ptr<ast::Fun> fun);
        return_type visit_impl(std::shared_ptr<ast::Main> main);
        return_type visit_impl(std::shared_ptr<ast::Selection> selection);
        return_type visit_impl(std::shared_ptr<ast::Variable> var);
        return_type visit_impl(std::shared_ptr<ast::Add> var);
        return_type visit_impl(std::shared_ptr<ast::Sub> sub);
        return_type visit_impl(std::shared_ptr<ast::Mul> mul);
        return_type visit_impl(std::shared_ptr<ast::Div> div1);
        return_type visit_impl(std::shared_ptr<ast::Mod> mod);
        return_type visit_impl(std::shared_ptr<ast::AggrSum> sum);
        return_type visit_impl(std::shared_ptr<ast::AggrCnt> cnt);
        return_type visit_impl(std::shared_ptr<ast::AggrMin> aggrMin);
        return_type visit_impl(std::shared_ptr<ast::AggrMax> aggrMax);
        return_type visit_impl(std::shared_ptr<ast::AggrAvg> avg);
        return_type visit_impl(std::shared_ptr<ast::Eq> eq);
        return_type visit_impl(std::shared_ptr<ast::Neq> neq);
        return_type visit_impl(std::shared_ptr<ast::Le> le);
        return_type visit_impl(std::shared_ptr<ast::Ge> ge);
        return_type visit_impl(std::shared_ptr<ast::Leq> leq);
        return_type visit_impl(std::shared_ptr<ast::Geq> geq);
        return_type visit_impl(std::shared_ptr<ast::And> anAnd);
        return_type visit_impl(std::shared_ptr<ast::Or> anOr);
        return_type visit_impl(std::shared_ptr<ast::Not> aNot);
        return_type visit_impl(std::shared_ptr<ast::Predicate> pred);
        return_type visit_impl(std::shared_ptr<ast::StatementWrapper> wrapper);
        return_type visit_impl(std::shared_ptr<ast::Hash> hash);
        return_type visit_impl(std::shared_ptr<ast::Lookup> lookup);
        return_type visit_impl(std::shared_ptr<ast::Insert> insert);
        return_type visit_impl(std::monostate);

        std::shared_ptr<Type> get_type(const ast::ASTNodeVariant &node) const;

        void set_arity(const ast::ASTNodeVariant &node, size_t ar);
        void set_type(const ast::ASTNodeVariant &node, DataType type);

        void insertNewType(const ast::ASTNodeVariant &node, DataType t, Arity ar);
        void insertNewTypeAs(const ast::ASTNodeVariant &node, const Type &t);

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
