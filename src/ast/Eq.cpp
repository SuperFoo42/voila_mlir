#include "ast/Eq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

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