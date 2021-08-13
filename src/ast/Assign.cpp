#include "ast/Assign.hpp"

namespace voila::ast
{
    Assign::Assign(Location loc, std::vector<Expression> dests, Statement expr) :
        IStatement(loc), dests{std::move(dests)}, expr{std::move(expr)}, pred{std::nullopt}
    {
        assert(ranges::all_of(this->dests, [](auto &dest) -> auto {return dest.is_variable() || dest.is_reference();}));
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