#include "ast/Comparison.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    bool Comparison::is_comparison() const { return true; }
    Comparison *Comparison::as_comparison() { return this; }
    std::string Comparison::type2string() const { return "comparison"; }
    void Comparison::print(std::ostream &) const {}
    Comparison::Comparison(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
        : IExpression(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
    {
    }
} // namespace voila::ast