#include <utility>

#include "ast/Assign.hpp"

namespace voila::ast
{
    Assign::Assign(const Location loc, std::string dest, Expression expr) :
        IStatement(loc), dest{std::move(dest)}, expr{std::move(expr)}, pred{std::nullopt}
    {
        // TODO: find dest variable and look for conflicts
    }
    Assign *Assign::as_assignment()
    {
        return this;
    }
    bool Assign::is_assignment() const
    {
        return true;
    }
    void Assign::predicate(Expression expression)
    {
        if (expression.is_predicate())
            pred = expression;
        else
            throw std::invalid_argument("Expression is no predicate");
    }
    void Assign::print(std::ostream &ostream) const
    {
        ostream << dest << " =";
    }

    void Assign::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Assign::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast