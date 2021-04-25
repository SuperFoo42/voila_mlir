#include "ast/Mul.hpp"

namespace voila::ast
{
    std::string Mul::type2string() const
    {
        return "mul";
    }
    bool Mul::is_mul() const
    {
        return true;
    }
    Mul *Mul::as_mul()
    {
        return this;
    }
    void Mul::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Mul::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast