#include "ast/And.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/Expression.hpp"  // for Expression

namespace voila::ast
{
    std::string And::type2string() const
    {
        return "and";
    }
    bool And::is_and() const
    {
        return true;
    }
    And *And::as_and()
    {
        return this;
    }
    void And::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void And::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> And::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_shared<And>(loc, mLhs.clone(vmap), mRhs.clone(vmap));
    }
} // namespace voila::ast