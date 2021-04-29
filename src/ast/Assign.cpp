#include "ast/Assign.hpp"

#include <cassert>
#include <utility>

namespace voila::ast
{
    Assign::Assign(const Location loc, Expression dest, Expression expr) :
        IStatement(loc), dest{std::move(dest)}, expr{std::move(expr)}, pred{std::nullopt}
    {
        assert(this->dest.is_variable() || this->dest.is_reference());
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
    void Assign::print(std::ostream &) const
    {

    }

    void Assign::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Assign::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
    std::string Assign::type2string() const
    {
        return "assignment";
    }
} // namespace voila::ast