#include "ast/Loop.hpp"
#include "ASTNodes.hpp"
#include "range/v3/all.hpp"

voila::ast::ASTNodeVariant voila::ast::Loop::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
{
    std::vector<ASTNodeVariant> clonedStmts;
    auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                   [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};

    return std::make_shared<Loop>(
        loc, std::visit(cloneVisitor, mPred),
        mStms | ranges::views::transform([&cloneVisitor](auto &item) { return std::visit(cloneVisitor, item); }));
}
