#include "ast/Mod.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Mod::type2string() const { return "mod"; }
    bool Mod::is_mod() const { return true; }
    Mod *Mod::as_mod() { return this; }
    ASTNodeVariant Mod::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Arithmetic::clone<Mod>(vmap);
    }
} // namespace voila::ast