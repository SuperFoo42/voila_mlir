#pragma once
#include "ASTNode.hpp"

#include <concepts>
#include <memory>

namespace voila::ast
{
    class ASTVisitor;

    class Expression
    {
        std::shared_ptr<ASTNode> mImpl;
        Expression(std::shared_ptr<ASTNode> impl) : mImpl{impl} {}

      public:
        template<typename ExprImpl, typename... Args>
        requires std::derived_from<ExprImpl, ASTNode> static Expression make(Args &&...args)
        {
            return Expression(std::shared_ptr<ExprImpl>(new ExprImpl(std::forward<Args>(args)...)));
        }

        friend std::ostream &operator<<(std::ostream &out, const Expression &t)
        {
            t.mImpl->print(out);
            return out;
        }

        bool is_get_pos() const
        {
            return mImpl->is_get_pos();
        }

        bool is_get_morsel() const
        {
            return mImpl->is_get_morsel();
        }

        bool is_aggr() const
        {
            return mImpl->is_aggr();
        }

        bool is_table_op() const
        {
            return mImpl->is_table_op();
        }

        bool is_constant() const
        {
            return mImpl->is_constant();
        }

        bool is_cast() const
        {
            return mImpl->is_cast();
        }

        bool is_select() const
        {
            return mImpl->is_select();
        }

        bool is_terminal() const
        {
            return mImpl->is_terminal();
        }

        bool is_tupleop() const
        {
            return mImpl->is_tupleop();
        }

        bool is_expr()
        {
            return mImpl->is_expr();
        }

        bool is_stmt() const
        {
            return mImpl->is_stmt();
        }

        bool is_arithmetic() const
        {
            return mImpl->is_arithmetic();
        }

        bool is_add() const
        {
            return mImpl->is_add();
        }

        bool is_sub() const
        {
            return mImpl->is_sub();
        }

        bool is_mul() const
        {
            return mImpl->is_mul();
        }

        bool is_div() const
        {
            return mImpl->is_div();
        }

        bool is_mod() const
        {
            return mImpl->is_mod();
        }

        bool is_comparison() const
        {
            return mImpl->is_comparison();
        }

        bool is_geq() const
        {
            return mImpl->is_geq();
        }

        bool is_ge() const
        {
            return mImpl->is_ge();
        }

        bool is_leq() const
        {
            return mImpl->is_leq();
        }

        bool is_le() const
        {
            return mImpl->is_le();
        }

        bool is_neq() const
        {
            return mImpl->is_neq();
        }

        bool is_eq() const
        {
            return mImpl->is_eq();
        }

        bool is_logical() const
        {
            return mImpl->is_logical();
        }

        bool is_unary() const
        {
            return mImpl->is_unary();
        }

        bool is_binary() const
        {
            return mImpl->is_binary();
        }

        bool is_and() const
        {
            return mImpl->is_and();
        }

        bool is_or() const
        {
            return mImpl->is_or();
        }

        bool is_not() const
        {
            return mImpl->is_not();
        }

        void visit(ASTVisitor &visitor)
        {
            mImpl->visit(visitor);
        }
        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
    };
} // namespace voila::ast