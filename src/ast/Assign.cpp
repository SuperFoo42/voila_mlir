#include "ast/Assign.hpp"
#include <cassert>                           // for assert
#include <stdexcept>                         // for invalid_argument
#include <utility>                           // for move
#include "ast/ASTVisitor.hpp"                // for ASTVisitor
#include "ast/Expression.hpp"                // for Expression
#include "ast/IStatement.hpp"                // for IStatement
#include "ast/Statement.hpp"                 // for Statement
#include "range/v3/algorithm/all_of.hpp"     // for all_of, all_of_fn
#include "range/v3/algorithm/transform.hpp"  // for transform, transform_fn
#include "range/v3/functional/identity.hpp"  // for identity

namespace voila::ast {
    Assign::Assign(Location loc, std::vector<Expression> dests, Statement expr) :
            IStatement(loc), pred{std::nullopt}, mDdests{std::move(dests)}, mExpr{std::move(expr)} {
        assert(ranges::all_of(this->mDdests,
                              [](auto &dest) -> auto { return dest.is_variable() || dest.is_reference(); }));
    }

    Assign *Assign::as_assignment() {
        return this;
    }

    bool Assign::is_assignment() const {
        return true;
    }

    void Assign::set_predicate(Expression expression) {
        if (expression.is_predicate())
            pred = expression;
        else
            throw std::invalid_argument("Expression is no predicate");
    }

    std::optional<Expression> Assign::get_predicate() {
        return pred;
    }

    void Assign::print(std::ostream &) const {

    }

    void Assign::visit(ASTVisitor &visitor) const {
        visitor(*this);
    }

    void Assign::visit(ASTVisitor &visitor) {
        visitor(*this);
    }

    std::string Assign::type2string() const {
        return "assignment";
    }

    std::shared_ptr<ASTNode> Assign::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        auto new_expr = mExpr.clone(vmap);
        std::vector<Expression> new_dests;
        ranges::transform(mDdests, new_dests.begin(), [&vmap](auto &val) { return val.clone(vmap); });

        auto clonedAssignment = std::make_shared<Assign>(loc, new_dests, new_expr);
        if (pred.has_value())
            clonedAssignment->pred = std::make_optional(pred->clone(vmap));

        return clonedAssignment;

    }
} // namespace voila::ast