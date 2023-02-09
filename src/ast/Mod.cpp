#include "ast/Mod.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    std::string Mod::type2string() const { return "mod"; }
    bool Mod::is_mod() const { return true; }
    Mod *Mod::as_mod() { return this; }
    void Mod::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Mod::visit(ASTVisitor &visitor) { visitor(*this); }
} // namespace voila::ast