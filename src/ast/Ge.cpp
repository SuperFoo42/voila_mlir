#include "ast/Ge.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Ge::type2string() const
    {
        return "ge";
    }
    bool Ge::is_ge() const
    {
        return true;
    }
    Ge *Ge::as_ge()
    {
        return this;
    }
    ASTNodeVariant Ge::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Ge>(vmap);
    }
} // namespace voila::ast