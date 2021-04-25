#include "ast/Read.hpp"

namespace voila::ast
{
    bool Read::is_read() const
    {
        return true;
    }
    Read *Read::as_read()
    {
        return this;
    }
    std::string Read::type2string() const
    {
        return "read";
    }
    void Read::print(std::ostream &) const {}
    void Read::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Read::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast