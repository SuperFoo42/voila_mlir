#include "ast/Hash.hpp"

namespace voila::ast
{
    [[nodiscard]] std::string Hash::type2string() const
    {
        return "hash";
    }

    [[nodiscard]] bool Hash::is_hash() const
    {
        return true;
    }

    Hash *Hash::as_hash()
    {
        return this;
    }
    void Hash::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Hash::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
    void Hash::print(std::ostream &) const {}
} // namespace voila::ast