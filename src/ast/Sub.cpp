#include "ast/Sub.hpp"

namespace voila::ast
{
    std::string Sub::type2string() const
    {
        return "sub";
    }
    bool Sub::is_sub() const
    {
        return true;
    }
    void Sub::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Sub::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast