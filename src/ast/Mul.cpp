#include "ast/Mul.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

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
    ASTNodeVariant Mul::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Arithmetic::clone<Mul>(vmap);
    }
} // namespace voila::ast