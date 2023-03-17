#include "ast/Assign.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "ast/IStatement.hpp"               // for IStatement
#include "range/v3/algorithm/all_of.hpp"    // for all_of, all_of_fn
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity
#include <cassert>                          // for assert
#include <stdexcept>                        // for invalid_argument
#include <utility>                          // for move

namespace voila::ast
{
    Assign::Assign(Location loc, std::vector<ASTNodeVariant> dests, ASTNodeVariant expr)
        : IStatement(loc), pred{}, mDdests{std::move(dests)}, mExpr{std::move(expr)}
    {
        assert(ranges::all_of(this->mDdests,
                              [](auto &dest) -> auto
                              {
                                  return std::visit(overloaded{[](auto) { return false; },
                                                               [](const std::shared_ptr<Variable> &) { return true; },
                                                               [](const std::shared_ptr<Ref> &) { return true; }},
                                                    dest);
                              }));
    }

    Assign *Assign::as_assignment() { return this; }

    bool Assign::is_assignment() const { return true; }

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

    void Assign::print(std::ostream &) const {}

    std::string Assign::type2string() const { return "assignment"; }

    ASTNodeVariant Assign::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { return std::monostate(); }};
        auto new_expr = std::visit(cloneVisitor, mExpr);

        std::vector<ASTNodeVariant> new_dests;
        ranges::transform(mDdests, new_dests.begin(),
                          [&cloneVisitor](auto &val) { return std::visit(cloneVisitor, val); });

        auto clonedAssignment = std::make_shared<Assign>(loc, new_dests, new_expr);

        clonedAssignment->pred = std::visit(cloneVisitor, pred);

        return clonedAssignment;
    }
} // namespace voila::ast