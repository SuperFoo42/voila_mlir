#include "ast/Or.hpp"

namespace voila::ast
{
    std::string Or::type2string() const
    {
        return "or";
    }
    bool Or::is_or() const
    {
        return true;
    }
    Or *Or::as_or()
    {
        return this;
    }
    void Or::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Or::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast