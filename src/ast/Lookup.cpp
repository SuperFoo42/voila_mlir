#include "ast/Lookup.hpp"
namespace voila::ast
{
    bool Lookup::is_lookup() const
    {
        return true;
    }
    Lookup *Lookup::as_lookup()
    {
        return this;
    }
    std::string Lookup::type2string() const
    {
        return "hash_insert";
    }
    void Lookup::print(std::ostream &) const {}
    void Lookup::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Lookup::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast