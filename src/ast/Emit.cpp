#include "ast/ASTNode.hpp"
#include "ast/Emit.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "range/v3/all.hpp"   // for transform, transform_fn

namespace voila::ast
{
    std::string Emit::type2string_impl() const { return "emit"; }

    void Emit::print_impl(std::ostream &) const {}

    ASTNodeVariant Emit::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Emit>(
            loc, mExprs | ranges::views::transform([&cloneVisitor](auto &el) { return std::visit(cloneVisitor, el); }));
    }
} // namespace voila::ast