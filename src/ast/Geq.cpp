#include "ast/Geq.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Geq::type2string() const
    {
        return "geq";
    }
    bool Geq::is_geq() const
    {
        return true;
    }
    Geq *Geq::as_geq()
    {
        return this;
    }
    ASTNodeVariant Geq::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        return Comparison::clone<Geq>(vmap);
    }
} // namespace voila::ast