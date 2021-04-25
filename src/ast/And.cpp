#include "ast/And.hpp"

namespace voila::ast
{
    std::string And::type2string() const
    {
        return "and";
    }
    bool And::is_and() const
    {
        return true;
    }
    And *And::as_and()
    {
        return this;
    }
    void And::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void And::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast