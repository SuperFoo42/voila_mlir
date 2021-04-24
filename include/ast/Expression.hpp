#pragma once
#include "IExpression.hpp"

#include <concepts>
#include <memory>
#include <utility>

namespace voila::ast
{
    class Expression
    {
        std::shared_ptr<IExpression> mImpl;
        explicit Expression(std::shared_ptr<IExpression> impl) : mImpl{std::move(impl)} {}

      public:
        Expression() = default;
        Expression(Expression &) = default;
        Expression(const Expression &) = default;
        Expression(Expression &&) = default;

        Expression &operator=(const Expression &) = default;

        template<typename ExprImpl, typename... Args>
        requires std::is_base_of_v<IExpression, ExprImpl>
        static Expression make(Args &&...args)
        {
            return Expression(std::make_shared<ExprImpl>(std::forward<Args>(args)...));
        }

        friend std::ostream &operator<<(std::ostream &out, const Expression &t);

        [[nodiscard]] bool is_expr() const;

        [[nodiscard]] bool is_select() const;

        [[nodiscard]] bool is_arithmetic() const;

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

        Ref *as_reference();

        [[nodiscard]] std::string type2string() const;

        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;

        void predicate(Expression expr);
        ;
        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
    };
} // namespace voila::ast