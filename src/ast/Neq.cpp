#include "ast/Neq.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Neq::type2string() const { return "neq"; }
    bool Neq::is_neq() const { return true; }
    Neq *Neq::as_neq() { return this; }
    ASTNodeVariant Neq::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Neq>(vmap);
    }
} // namespace voila::ast