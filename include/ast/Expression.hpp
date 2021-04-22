#pragma once
#include "IExpression.hpp"

#include <concepts>
#include <memory>

namespace voila::ast
{
    class Expression
    {
        std::shared_ptr<IExpression> mImpl;
        explicit Expression(std::shared_ptr<IExpression> impl) : mImpl{impl} {}

      public:
        template<typename ExprImpl, typename... Args>
        requires std::derived_from<ExprImpl, IExpression> static Expression make(Args &&...args)
        {
            return Expression(std::shared_ptr<ExprImpl>(new ExprImpl(std::forward<Args>(args)...)));
        }

        friend std::ostream &operator<<(std::ostream &out, const Expression &t)
        {
            t.mImpl->print(out);
            return out;
        }

        bool is_expr() const
        {
            return true;
        }

        bool is_select() const
        {
            return mImpl->is_select();
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

        bool is_string() const
        {
            return mImpl->is_string();
        }

        bool is_float() const
        {
            return mImpl->is_float();
        }

        bool is_integer() const
        {
            return mImpl->is_integer();
        }

        bool is_bool() const
        {
            return mImpl->is_bool();
        }

        bool is_const() const
        {
            return mImpl->is_const();
        }

        bool is_read() const
        {
            return mImpl->is_read();
        }

        bool is_gather() const
        {
            return mImpl->is_gather();
        }

        bool is_tuple_get() const
        {
            return mImpl->is_tuple_get();
        }

        bool is_reference() const
        {
            return mImpl->is_reference();
        }

        // casts
        IExpression *as_expr() const
        {
            return mImpl->as_expr();
        }

        Selection *as_select() const
        {
            return mImpl->as_select();
        }

        Arithmetic *as_arithmetic() const
        {
            return mImpl->as_arithmetic();
        }

        Add *as_add() const
        {
            return mImpl->as_add();
        }

        Sub *as_sub() const
        {
            return mImpl->as_sub();
        }

        Mul *as_mul() const
        {
            return mImpl->as_mul();
        }

        Div *as_div() const
        {
            return mImpl->as_div();
        }

        Mod *as_mod() const
        {
            return mImpl->as_mod();
        }

        Comparison *as_comparison() const
        {
            return mImpl->as_comparison();
        }

        Geq *as_geq() const
        {
            return mImpl->as_geq();
        }

        Ge *as_ge() const
        {
            return mImpl->as_ge();
        }

        Leq *as_leq() const
        {
            return mImpl->as_leq();
        }

        Le *as_le() const
        {
            return mImpl->as_le();
        }

        Neq *as_neq() const
        {
            return mImpl->as_neq();
        }

        Eq *as_eq() const
        {
            return mImpl->as_eq();
        }

        Logical *as_logical() const
        {
            return mImpl->as_logical();
        }

        And *as_and() const
        {
            return mImpl->as_and();
        }

        Or *as_or() const
        {
            return mImpl->as_or();
        }

        Not *as_not() const
        {
            return mImpl->as_not();
        }

        StrConst *as_string() const
        {
            return mImpl->as_string();
        }

        FltConst *as_float() const
        {
            return mImpl->as_float();
        }

        IntConst *as_integer() const
        {
            return mImpl->as_integer();
        }

        BooleanConst *as_bool() const
        {
            return mImpl->as_bool();
        }

        Const *as_const() const
        {
            return mImpl->as_const();
        }

        Read *as_read() const
        {
            return mImpl->as_read();
        }

        Gather *as_gather() const
        {
            return mImpl->as_gather();
        }

        TupleGet *as_tuple_get() const
        {
            return mImpl->as_tuple_get();
        }

        Ref *as_reference()
        {
            return mImpl->as_reference();
        }

        std::string type2string() const
        {
            return mImpl->type2string();
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