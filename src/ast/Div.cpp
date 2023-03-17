#include "ast/Div.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Div::type2string() const
    {
        return "div";
    }
    bool Div::is_div() const
    {
        return true;
    }
    Div *Div::as_div()
    {
        return this;
    }
    ASTNodeVariant Div::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Arithmetic::clone<Div>(vmap);
    }
} // namespace voila::ast