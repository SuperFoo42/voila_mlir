#include "ast/Selection.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp" // for Expression

namespace voila::ast
{
    bool Selection::is_select() const { return true; }
    Selection *Selection::as_select() { return this; }
    std::string Selection::type2string() const { return "selection"; }
    void Selection::print(std::ostream &) const {}
    void Selection::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Selection::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Selection::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        return std::make_shared<Selection>(loc, mParam.clone(vmap), mPred.clone(vmap));
    }
} // namespace voila::ast