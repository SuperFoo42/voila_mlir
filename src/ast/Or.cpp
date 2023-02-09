#include "ast/Or.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp" // for Expression

namespace voila::ast
{
    std::string Or::type2string() const { return "or"; }
    bool Or::is_or() const { return true; }
    Or *Or::as_or() { return this; }
    void Or::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Or::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Or::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        return std::make_shared<Or>(loc, mLhs.clone(vmap), mRhs.clone(vmap));
    }
} // namespace voila::ast