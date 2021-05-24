#include "ast/Expression.hpp"
namespace voila::ast
{
    std::ostream &operator<<(std::ostream &out, const Expression &t)
    {
        t.mImpl->print(out);
        return out;
    }
    bool Expression::is_expr() const
    {
        return true;
    }
    bool Expression::is_select() const
    {
        return mImpl->is_select();
    }
    bool Expression::is_arithmetic() const
    {
        return mImpl->is_arithmetic();
    }
    bool Expression::is_add() const
    {
        return mImpl->is_add();
    }
    bool Expression::is_sub() const
    {
        return mImpl->is_sub();
    }
    bool Expression::is_mul() const
    {
        return mImpl->is_mul();
    }
    bool Expression::is_div() const
    {
        return mImpl->is_div();
    }
    bool Expression::is_mod() const
    {
        return mImpl->is_mod();
    }
    bool Expression::is_comparison() const
    {
        return mImpl->is_comparison();
    }
    bool Expression::is_geq() const
    {
        return mImpl->is_geq();
    }
    bool Expression::is_ge() const
    {
        return mImpl->is_ge();
    }
    bool Expression::is_leq() const
    {
        return mImpl->is_leq();
    }
    bool Expression::is_le() const
    {
        return mImpl->is_le();
    }
    bool Expression::is_neq() const
    {
        return mImpl->is_neq();
    }
    bool Expression::is_eq() const
    {
        return mImpl->is_eq();
    }
    bool Expression::is_logical() const
    {
        return mImpl->is_logical();
    }
    bool Expression::is_unary() const
    {
        return mImpl->is_unary();
    }
    bool Expression::is_binary() const
    {
        return mImpl->is_binary();
    }
    bool Expression::is_and() const
    {
        return mImpl->is_and();
    }
    bool Expression::is_or() const
    {
        return mImpl->is_or();
    }
    bool Expression::is_not() const
    {
        return mImpl->is_not();
    }
    bool Expression::is_string() const
    {
        return mImpl->is_string();
    }
    bool Expression::is_float() const
    {
        return mImpl->is_float();
    }
    bool Expression::is_integer() const
    {
        return mImpl->is_integer();
    }
    bool Expression::is_bool() const
    {
        return mImpl->is_bool();
    }
    bool Expression::is_const() const
    {
        return mImpl->is_const();
    }
    bool Expression::is_read() const
    {
        return mImpl->is_read();
    }
    bool Expression::is_gather() const
    {
        return mImpl->is_gather();
    }
    bool Expression::is_tuple_get() const
    {
        return mImpl->is_tuple_get();
    }
    bool Expression::is_reference() const
    {
        return mImpl->is_reference();
    }
    IExpression *Expression::as_expr() const
    {
        return mImpl->as_expr();
    }
    Selection *Expression::as_select() const
    {
        return mImpl->as_select();
    }
    Arithmetic *Expression::as_arithmetic() const
    {
        return mImpl->as_arithmetic();
    }
    Add *Expression::as_add() const
    {
        return mImpl->as_add();
    }
    Sub *Expression::as_sub() const
    {
        return mImpl->as_sub();
    }
    Mul *Expression::as_mul() const
    {
        return mImpl->as_mul();
    }
    Div *Expression::as_div() const
    {
        return mImpl->as_div();
    }
    Mod *Expression::as_mod() const
    {
        return mImpl->as_mod();
    }
    Comparison *Expression::as_comparison() const
    {
        return mImpl->as_comparison();
    }
    Geq *Expression::as_geq() const
    {
        return mImpl->as_geq();
    }
    Ge *Expression::as_ge() const
    {
        return mImpl->as_ge();
    }
    Leq *Expression::as_leq() const
    {
        return mImpl->as_leq();
    }
    Le *Expression::as_le() const
    {
        return mImpl->as_le();
    }
    Eq *Expression::as_eq() const
    {
        return mImpl->as_eq();
    }
    Neq *Expression::as_neq() const
    {
        return mImpl->as_neq();
    }
    And *Expression::as_and() const
    {
        return mImpl->as_and();
    }
    Logical *Expression::as_logical() const
    {
        return mImpl->as_logical();
    }
    Not *Expression::as_not() const
    {
        return mImpl->as_not();
    }
    Or *Expression::as_or() const
    {
        return mImpl->as_or();
    }
    StrConst *Expression::as_string() const
    {
        return mImpl->as_string();
    }
    FltConst *Expression::as_float() const
    {
        return mImpl->as_float();
    }
    IntConst *Expression::as_integer() const
    {
        return mImpl->as_integer();
    }
    BooleanConst *Expression::as_bool() const
    {
        return mImpl->as_bool();
    }
    Const *Expression::as_const() const
    {
        return mImpl->as_const();
    }
    Read *Expression::as_read() const
    {
        return mImpl->as_read();
    }
    Gather *Expression::as_gather() const
    {
        return mImpl->as_gather();
    }
    TupleGet *Expression::as_tuple_get() const
    {
        return mImpl->as_tuple_get();
    }
    const Ref *Expression::as_reference() const
    {
        return mImpl->as_reference();
    }
    std::string Expression::type2string() const
    {
        return mImpl->type2string();
    }
    void Expression::visit(ASTVisitor &visitor)
    {
        mImpl->visit(visitor);
    }
    void Expression::set_predicate(Expression expr)
    {
        mImpl->set_predicate(std::move(expr));
    }
    void Expression::visit(ASTVisitor &visitor) const {
        mImpl->visit(visitor);
    }
    Predicate *Expression::as_predicate() const
    {
        return mImpl->as_predicate();
    }
    bool Expression::is_predicate() const
    {
        return mImpl->is_predicate();
    }
    Location Expression::get_location()
    {
        return mImpl->get_location();
    }
    Variable *Expression::as_variable() const
    {
        return mImpl->as_variable();
    }
    bool Expression::is_variable() const
    {
        return mImpl->is_variable();
    }

    std::optional<Expression> Expression::get_predicate() const
    {
        return mImpl->get_predicate();
    }
} // namespace voila::ast