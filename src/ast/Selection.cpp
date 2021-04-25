#include "ast/Selection.hpp"

namespace voila::ast
{
    bool Selection::is_select() const
    {
        return true;
    }
    Selection *Selection::as_select()
    {
        return this;
    }
    std::string Selection::type2string() const
    {
        return "selection";
    }
    void Selection::print(std::ostream &) const {}
    void Selection::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Selection::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast