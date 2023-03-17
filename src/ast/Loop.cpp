#include "ast/Loop.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    std::string Loop::type2string() const { return "loop"; }
    Loop *Loop::as_loop() { return this; }
    bool Loop::is_loop() const { return true; }
    void Loop::print(std::ostream &) const {}

    ASTNodeVariant Loop::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedStmts;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        ranges::transform(mStms, clonedStmts.begin(),
                          [&cloneVisitor](auto &item) { return std::visit(cloneVisitor, item); });
        return std::make_shared<Loop>(loc, std::visit(cloneVisitor, mPred), clonedStmts);
    }
} // namespace voila::ast