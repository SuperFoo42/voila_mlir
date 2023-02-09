#include "ast/Div.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    std::string Div::type2string() const
    {
        return "div";
    }
    bool Div::is_div() const
    {
        return true;
    }
    Div *Div::as_div()
    {
        return this;
    }
    void Div::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Div::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast