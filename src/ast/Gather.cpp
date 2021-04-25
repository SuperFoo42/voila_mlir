#include "ast/Gather.hpp"

namespace voila::ast
{
    bool Gather::is_gather() const
    {
        return true;
    }
    Gather *Gather::as_gather()
    {
        return this;
    }
    std::string Gather::type2string() const
    {
        return "gather";
    }
    void Gather::print(std::ostream &) const {}
    void Gather::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Gather::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast