#include "ast/Write.hpp"
#include "ASTNodes.hpp"

void voila::ast::Write::print_impl(std::ostream &ostream) const

{
    std::visit(overloaded{[&ostream](auto &src) -> void { ostream << "src: " << *src; }, [](std::monostate) {}}, mSrc);
    std::visit(overloaded{[&ostream](auto &dest) -> void { ostream << "dest: " << *dest; }, [](std::monostate) {}}, mDest);
}
voila::ast::ASTNodeVariant voila::ast::Write::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
{
    auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                   [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
    return std::make_shared<Write>(loc, std::visit(cloneVisitor, mSrc), std::visit(cloneVisitor, mDest),
                                   std::visit(cloneVisitor, mStart));
}
