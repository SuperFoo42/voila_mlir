#include "ast/Arithmetic.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/IExpression.hpp" // for IExpression
#include <utility>             // for move

namespace voila::ast
{
    std::string Arithmetic::type2string() const { return "arithmetic"; }
    Arithmetic *Arithmetic::as_arithmetic() { return this; }
    bool Arithmetic::is_arithmetic() const { return true; }
    void Arithmetic::print(std::ostream &) const {}

    Arithmetic::Arithmetic(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
        : IExpression(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
    {
    }
} // namespace voila::ast