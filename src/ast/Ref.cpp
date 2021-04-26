#include "ast/Ref.hpp"

namespace voila::ast
{
    Ref::Ref(const  Location loc, const std::string &var) : IExpression(loc)
    {
        (void)var;
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
    void Ref::print(std::ostream &) const
    {
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