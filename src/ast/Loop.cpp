#include "ast/Loop.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "ast/Expression.hpp"               // for Expression
#include "ast/Statement.hpp"                // for Statement
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    std::string Loop::type2string() const { return "loop"; }
    Loop *Loop::as_loop() { return this; }
    bool Loop::is_loop() const { return true; }
    void Loop::print(std::ostream &) const {}
    void Loop::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Loop::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Loop::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        std::vector<Statement> clonedStmts;
        ranges::transform(mStms, clonedStmts.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        return std::make_shared<Loop>(loc, mPred.clone(vmap), clonedStmts);
    }
} // namespace voila::ast