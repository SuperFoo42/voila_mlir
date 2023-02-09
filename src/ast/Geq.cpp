#include "ast/Geq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    std::string Geq::type2string() const
    {
        return "geq";
    }
    bool Geq::is_geq() const
    {
        return true;
    }
    Geq *Geq::as_geq()
    {
        return this;
    }
    void Geq::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Geq::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast