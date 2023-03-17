#include "ast/Le.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Le::type2string() const
    {
        return "le";
    }
    bool Le::is_le() const
    {
        return true;
    }
    Le *Le::as_le()
    {
        return this;
    }
    ASTNodeVariant Le::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Le>(vmap);
    }
} // namespace voila::ast