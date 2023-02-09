#include "ast/Le.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    std::string Le::type2string() const
    {
        return "le";
    }
    bool Le::is_le() const
    {
        return true;
    }
    Le *Le::as_le()
    {
        return this;
    }
    void Le::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Le::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast