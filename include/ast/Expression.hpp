#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr, dynamic_pointer_cast, swap
#include <optional>             // for optional
#include <string>               // for string
#include <utility>              // for move, forward
#include "ast/ASTNode.hpp"      // for ASTNode, Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap
#include "ast/IExpression.hpp"

namespace voila::ast {
    class Selection;
    class Const;
    class Add;
    class Arithmetic;
    class Sub;
    class Mul;
    class Div;
    class Mod;
    class Comparison;
    class Eq;
    class Neq;
    class Le;
    class Ge;
    class Leq;
    class Geq;
    class And;
    class Or;
    class Not;
    class Logical;
    class IntConst;
    class BooleanConst;
    class FltConst;
    class StrConst;
    class Read;
    class Gather;
    class Ref;
    class TupleGet;
    class TupleCreate;
    class Expression;
    class Predicate;
    class Variable;
    class Aggregation;
    class AggrSum;
    class AggrCnt;
    class AggrMin;
    class AggrMax;
    class AggrAvg;
    class Hash;
    class Lookup;
    class Insert;
    class Scatter;

    class ASTVisitor;

    class Expression {
        std::shared_ptr<IExpression> mImpl;

        explicit Expression(std::shared_ptr<IExpression> impl) : mImpl{std::move(impl)} {}

        explicit Expression(std::unique_ptr<ASTNode> &&impl) {
            if (impl->is_expr())
                mImpl = std::dynamic_pointer_cast<IExpression>(std::shared_ptr<ASTNode>(std::move(impl)));
            else
                throw; //TODO type mismatch exception
        }


    public:
        Expression() = default;

        Expression(Expression &) = default;

        Expression(const Expression &) = default;

        Expression(Expression &&) = default;

        Expression &operator=(const Expression &expr) {
            Expression(expr).swap(*this);
            return *this;
        }

        void swap(Expression &s) noexcept
        {
            std::swap(this->mImpl, s.mImpl);
        }

        template<typename ExprImpl, typename... Args>
        requires std::is_base_of_v<IExpression, ExprImpl>
        static Expression make(Args &&...args) {
            return Expression(std::make_shared<ExprImpl>(std::forward<Args>(args)...));
        }

        template<typename ExprImpl>
        requires std::is_base_of_v<IExpression, ExprImpl>
        static Expression make(std::shared_ptr<ExprImpl> ptr) {
            return Expression(ptr);
        }

        friend std::ostream &operator<<(std::ostream &out, const Expression &t);

        [[nodiscard]] bool is_expr() const;

        [[nodiscard]] bool is_select() const;

        [[nodiscard]] bool is_arithmetic() const;

        [[nodiscard]] bool is_scatter() const;

        [[nodiscard]] bool is_add() const;

        [[nodiscard]] bool is_sub() const;

        [[nodiscard]] bool is_mul() const;

        [[nodiscard]] bool is_div() const;

        [[nodiscard]] bool is_mod() const;

        [[nodiscard]] bool is_comparison() const;

        [[nodiscard]] bool is_geq() const;

        [[nodiscard]] bool is_ge() const;

        [[nodiscard]] bool is_leq() const;

        [[nodiscard]] bool is_le() const;

        [[nodiscard]] bool is_neq() const;

        [[nodiscard]] bool is_eq() const;

        [[nodiscard]] bool is_logical() const;

        [[nodiscard]] bool is_unary() const;

        [[nodiscard]] bool is_binary() const;

        [[nodiscard]] bool is_and() const;

        [[nodiscard]] bool is_or() const;

        [[nodiscard]] bool is_not() const;

        [[nodiscard]] bool is_string() const;

        [[nodiscard]] bool is_float() const;

        [[nodiscard]] bool is_integer() const;

        [[nodiscard]] bool is_bool() const;

        [[nodiscard]] bool is_const() const;

        [[nodiscard]] bool is_read() const;

        [[nodiscard]] bool is_gather() const;

        [[nodiscard]] bool is_tuple_get() const;

        [[nodiscard]] bool is_reference() const;

        [[nodiscard]] bool is_predicate() const;

        [[nodiscard]] bool is_variable() const;

        [[nodiscard]] bool is_aggr() const;

        [[nodiscard]] bool is_aggr_sum() const;

        [[nodiscard]] bool is_aggr_cnt() const;

        [[nodiscard]] bool is_aggr_min() const;

        [[nodiscard]] bool is_aggr_max() const;

        [[nodiscard]] bool is_aggr_avg() const;

        [[nodiscard]] bool is_hash() const;

        [[nodiscard]] bool is_lookup() const;

        // casts
        [[nodiscard]] IExpression *as_expr() const;

        [[nodiscard]] Selection *as_select() const;

        [[nodiscard]] Arithmetic *as_arithmetic() const;

        [[nodiscard]] Add *as_add() const;

        [[nodiscard]] Sub *as_sub() const;

        [[nodiscard]] Mul *as_mul() const;

        [[nodiscard]] Div *as_div() const;

        [[nodiscard]] Mod *as_mod() const;

        [[nodiscard]] Comparison *as_comparison() const;

        [[nodiscard]] Geq *as_geq() const;

        [[nodiscard]] Ge *as_ge() const;

        [[nodiscard]] Leq *as_leq() const;

        [[nodiscard]] Le *as_le() const;

        [[nodiscard]] Neq *as_neq() const;

        [[nodiscard]] Eq *as_eq() const;

        [[nodiscard]] Logical *as_logical() const;

        [[nodiscard]] And *as_and() const;

        [[nodiscard]] Or *as_or() const;

        [[nodiscard]] Not *as_not() const;

        [[nodiscard]] StrConst *as_string() const;

        [[nodiscard]] FltConst *as_float() const;

        [[nodiscard]] IntConst *as_integer() const;

        [[nodiscard]] BooleanConst *as_bool() const;

        [[nodiscard]] Const *as_const() const;

        [[nodiscard]] Read *as_read() const;

        [[nodiscard]] Gather *as_gather() const;

        [[nodiscard]] TupleGet *as_tuple_get() const;

        [[nodiscard]] const Ref *as_reference() const;

        [[nodiscard]] Predicate *as_predicate() const;

        [[nodiscard]] Variable *as_variable() const;

        [[nodiscard]] Aggregation *as_aggr() const;

        [[nodiscard]] AggrSum *as_aggr_sum() const;

        [[nodiscard]] AggrCnt *as_aggr_cnt() const;

        [[nodiscard]] AggrMin *as_aggr_min() const;

        [[nodiscard]] AggrMax *as_aggr_max() const;

        [[nodiscard]] AggrAvg *as_aggr_avg() const;

        [[nodiscard]] Hash *as_hash() const;

        [[nodiscard]] Lookup *as_lookup() const;

        [[nodiscard]] bool is_insert() const;

        [[nodiscard]] Insert *as_insert() const;

        [[nodiscard]] Scatter *as_scatter() const;

        [[nodiscard]] std::string type2string() const;

        void visit(ASTVisitor &visitor);

        void visit(ASTVisitor &visitor) const;

        [[nodiscard]] Expression clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap);

        void set_predicate(Expression expr);

        [[nodiscard]] std::optional<Expression> get_predicate() const;

        Location get_location();

        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
    };
} // namespace voila::ast