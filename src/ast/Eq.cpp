#include "ast/Eq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Eq::type2string() const
    {
        return "eq";
    }
    bool Eq::is_eq() const
    {
        return true;
    }
    Eq *Eq::as_eq()
    {
        return this;
    }
    ASTNodeVariant Eq::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Eq>(vmap);
    }
} // namespace voila::ast