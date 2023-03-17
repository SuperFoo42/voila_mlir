#include "ast/Sub.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Sub::type2string() const
    {
        return "sub";
    }
    bool Sub::is_sub() const
    {
        return true;
    }
    ASTNodeVariant Sub::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Arithmetic::clone<Sub>(vmap);
    }
} // namespace voila::ast