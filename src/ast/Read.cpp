#include "ast/Read.hpp"
#include "ASTNodes.hpp"

voila::ast::ASTNodeVariant voila::ast::Read::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
{
    auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                   [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
    return std::make_shared<Read>(loc, std::visit(cloneVisitor, mColumn), std::visit(cloneVisitor, mIdx));
}
