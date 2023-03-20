#include "ast/Assign.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/functional/identity.hpp" // for identity
#include "range/v3/view/transform.hpp"      // for transform, transform_fn
#include <cassert>                          // for assert
#include <stdexcept>                        // for invalid_argument
#include <utility>                          // for move

namespace voila::ast
{
    void Assign::set_predicate(ASTNodeVariant expression)
    {
        pred = std::visit(overloaded{[](std::shared_ptr<Predicate> &p) -> ASTNodeVariant { return p; },
                                     [](auto) -> ASTNodeVariant
                                     { throw std::invalid_argument("Expression is no predicate"); }},
                          expression);
    }

    std::optional<ASTNodeVariant> Assign::get_predicate()
    {
        return std::visit(overloaded{[](std::monostate) -> std::optional<ASTNodeVariant> { return std::nullopt; },
                                     [](auto &e) -> std::optional<ASTNodeVariant> { return ASTNodeVariant(e); }},
                          pred);
    }

    void Assign::print_impl(std::ostream &) const {}

    std::string Assign::type2string_impl() const { return "assignment"; }

    ASTNodeVariant Assign::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        auto cvTransform =
            ranges::views::transform([&cloneVisitor](auto &arg) { return std::visit(cloneVisitor, arg); });
        auto new_expr = std::visit(cloneVisitor, mExpr);

        auto clonedAssignment = std::make_shared<Assign>(loc, mDdests | cvTransform, new_expr);

        clonedAssignment->pred = std::visit(cloneVisitor, pred);

        return clonedAssignment;
    }
} // namespace voila::ast