#include "ast/IExpression.hpp"

#include "ast/Expression.hpp"

namespace voila::ast
{
    void IExpression::predicate(Expression)
    {
        throw PredicationUnsupportedException(fmt::format("Predication not supported for: {}.", type2string()));
    }
    bool IExpression::is_expr() const
    {
        return true;
    }
    bool IExpression::is_select() const
    {
        return false;
    }
    bool IExpression::is_arithmetic() const
    {
        return false;
    }
    bool IExpression::is_sub() const
    {
        return false;
    }
    bool IExpression::is_add() const
    {
        return false;
    }
    bool IExpression::is_div() const
    {
        return false;
    }
    bool IExpression::is_mul() const
    {
        return false;
    }
    bool IExpression::is_mod() const
    {
        return false;
    }
    bool IExpression::is_comparison() const
    {
        return false;
    }
    bool IExpression::is_geq() const
    {
        return false;
    }
    bool IExpression::is_ge() const
    {
        return false;
    }
    bool IExpression::is_leq() const
    {
        return false;
    }
    bool IExpression::is_le() const
    {
        return false;
    }
    bool IExpression::is_neq() const
    {
        return false;
    }
    bool IExpression::is_eq() const
    {
        return false;
    }
    bool IExpression::is_logical() const
    {
        return false;
    }
    bool IExpression::is_unary() const
    {
        return false;
    }
    bool IExpression::is_binary() const
    {
        return false;
    }
    bool IExpression::is_and() const
    {
        return false;
    }
    bool IExpression::is_or() const
    {
        return false;
    }
    bool IExpression::is_not() const
    {
        return false;
    }
    bool IExpression::is_string() const
    {
        return false;
    }
    bool IExpression::is_float() const
    {
        return false;
    }
    bool IExpression::is_integer() const
    {
        return false;
    }
    bool IExpression::is_bool() const
    {
        return false;
    }
    bool IExpression::is_const() const
    {
        return false;
    }
    bool IExpression::is_read() const
    {
        return false;
    }
    bool IExpression::is_gather() const
    {
        return false;
    }
    bool IExpression::is_tuple_get() const
    {
        return false;
    }
    bool IExpression::is_reference() const
    {
        return false;
    }
    bool IExpression::is_tuple_create() const
    {
        return false;
    }
    IExpression *IExpression::as_expr()
    {
        return this;
    }
    Selection *IExpression::as_select()
    {
        return nullptr;
    }
    Arithmetic *IExpression::as_arithmetic()
    {
        return nullptr;
    }
    Add *IExpression::as_add()
    {
        return nullptr;
    }
    Sub *IExpression::as_sub()
    {
        return nullptr;
    }
    Mul *IExpression::as_mul()
    {
        return nullptr;
    }
    Div *IExpression::as_div()
    {
        return nullptr;
    }
    Mod *IExpression::as_mod()
    {
        return nullptr;
    }
    Comparison *IExpression::as_comparison()
    {
        return nullptr;
    }
    Geq *IExpression::as_geq()
    {
        return nullptr;
    }
    Ge *IExpression::as_ge()
    {
        return nullptr;
    }
    Leq *IExpression::as_leq()
    {
        return nullptr;
    }
    Le *IExpression::as_le()
    {
        return nullptr;
    }
    Neq *IExpression::as_neq()
    {
        return nullptr;
    }
    Eq *IExpression::as_eq()
    {
        return nullptr;
    }
    Logical *IExpression::as_logical()
    {
        return nullptr;
    }
    And *IExpression::as_and()
    {
        return nullptr;
    }
    Or *IExpression::as_or()
    {
        return nullptr;
    }
    Not *IExpression::as_not()
    {
        return nullptr;
    }
    StrConst *IExpression::as_string()
    {
        return nullptr;
    }
    FltConst *IExpression::as_float()
    {
        return nullptr;
    }
    IntConst *IExpression::as_integer()
    {
        return nullptr;
    }
    BooleanConst *IExpression::as_bool()
    {
        return nullptr;
    }
    Const *IExpression::as_const()
    {
        return nullptr;
    }
    Read *IExpression::as_read()
    {
        return nullptr;
    }
    Gather *IExpression::as_gather()
    {
        return nullptr;
    }
    TupleGet *IExpression::as_tuple_get()
    {
        return nullptr;
    }
    Ref *IExpression::as_reference()
    {
        return nullptr;
    }
    TupleCreate *IExpression::as_tuple_create()
    {
        return nullptr;
    }
} // namespace voila::ast