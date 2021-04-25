#include "ast/Eq.hpp"

namespace voila::ast
{
    std::string Eq::type2string() const
    {
        return "eq";
    }
    bool Eq::is_eq() const
    {
        return true;
    }
    Eq *Eq::as_eq()
    {
        return this;
    }
    void Eq::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Eq::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast