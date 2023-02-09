#include "ast/Gather.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ast/Expression.hpp"  // for Expression

namespace voila::ast
{
    bool Gather::is_gather() const
    {
        return true;
    }
    Gather *Gather::as_gather()
    {
        return this;
    }
    std::string Gather::type2string() const
    {
        return "gather";
    }
    void Gather::print(std::ostream &) const {}
    void Gather::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Gather::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> Gather::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_shared<Gather>(loc, mColumn.clone(vmap), mIdxs.clone(vmap));
    }
} // namespace voila::ast