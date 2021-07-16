#include "ast/Insert.hpp"

namespace voila::ast
{
    bool Insert::is_insert() const
    {
        return true;
    }
    Insert *Insert::as_insert()
    {
        return this;
    }
    std::string Insert::type2string() const
    {
        return "hash_insert";
    }
    void Insert::print(std::ostream &) const {

    }
    void Insert::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Insert::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast