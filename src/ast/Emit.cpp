#include "ast/Emit.hpp"

namespace voila::ast
{
    bool Emit::is_emit() const
    {
        return true;
    }

    Emit *Emit::as_emit()
    {
        return this;
    }

    std::string Emit::type2string() const
    {
        return "emit";
    }

    void Emit::print(std::ostream &) const
    {
    }
    void Emit::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Emit::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast