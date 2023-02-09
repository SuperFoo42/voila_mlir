#include "ast/Arithmetic.hpp"
#include <utility>              // for move
#include "ast/Expression.hpp"   // for Expression
#include "ast/IExpression.hpp"  // for IExpression

namespace voila::ast
{
    std::string Arithmetic::type2string() const
    {
        return "arithmetic";
    }
    Arithmetic *Arithmetic::as_arithmetic()
    {
        return this;
    }
    bool Arithmetic::is_arithmetic() const
    {
        return true;
    }
    void Arithmetic::print(std::ostream &) const {}

    Arithmetic::Arithmetic(const Location loc, Expression lhs, Expression rhs) :
        IExpression(loc), mLhs {std::move(lhs)}, mRhs{std::move(rhs)} {}
} // namespace voila::ast