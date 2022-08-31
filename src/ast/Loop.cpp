#include "ast/Loop.hpp"
#include "range/v3/algorithm.hpp"
namespace voila::ast
{
    std::string Loop::type2string() const
    {
        return "loop";
    }
    Loop *Loop::as_loop()
    {
        return this;
    }
    bool Loop::is_loop() const
    {
        return true;
    }
    void Loop::print(std::ostream &) const
    {
    }
    void Loop::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Loop::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Loop::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Statement> clonedStmts;
        ranges::transform(stms, clonedStmts.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        return std::make_unique<Loop>(loc, pred.clone(vmap), clonedStmts);
    }
} // namespace voila::ast