#include "ast/Ref.hpp"
#include "ast/Expression.hpp"
#include "ast/Variable.hpp"

namespace voila::ast
{
    Ref::Ref(const Location loc, Expression var) : IExpression(loc), ref{std::move(var)}
    {
        // TODO find reference or error
    }
    bool Ref::is_reference() const
    {
        return true;
    }
    std::string Ref::type2string() const
    {
        return "reference";
    }
    Ref *Ref::as_reference()
    {
        return this;
    }
    void Ref::print(std::ostream &ostream) const
    {
        ostream << ref.as_variable()->var;
    }

    void Ref::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Ref::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast