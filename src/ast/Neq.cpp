#include "ast/Neq.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    std::string Neq::type2string() const { return "neq"; }
    bool Neq::is_neq() const { return true; }
    Neq *Neq::as_neq() { return this; }
    void Neq::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Neq::visit(ASTVisitor &visitor) { visitor(*this); }
} // namespace voila::ast