#include "ast/Ge.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    std::string Ge::type2string() const
    {
        return "ge";
    }
    bool Ge::is_ge() const
    {
        return true;
    }
    Ge *Ge::as_ge()
    {
        return this;
    }
    void Ge::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Ge::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast