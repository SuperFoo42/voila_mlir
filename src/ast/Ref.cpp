#include "ast/Ref.hpp"
#include "ASTNodes.hpp"

void voila::ast::Ref::print_impl(std::ostream &ostream) const
{
    std::visit(overloaded{[&ostream](std::shared_ptr<Variable> &var) { ostream << var->var; }, [](auto) {}}, mRef);
}
voila::ast::ASTNodeVariant voila::ast::Ref::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
{
    auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                   [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};;
    return std::make_shared<Ref>(loc, std::visit(cloneVisitor, mRef));
}
