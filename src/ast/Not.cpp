#include "ast/Not.hpp"

namespace voila::ast
{
    std::string Not::type2string() const
    {
        return "not";
    }
    bool Not::is_not() const
    {
        return true;
    }
    Not *Not::as_not()
    {
        return this;
    }

    void Not::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Not::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast