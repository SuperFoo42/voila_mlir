#include "ast/Assign.hpp"

#include <ast/Statement.hpp>
#include <cassert>
#include <utility>

namespace voila::ast
{
    Assign::Assign(Location loc, Expression dest, Statement expr) :
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
    void Assign::set_predicate(Expression expression)
    {
        if (expression.is_predicate())
            pred = expression;
        else
            throw std::invalid_argument("Expression is no predicate");
    }

    std::optional<Expression> Assign::get_predicate()
    {
        return pred;
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