#include "ast/Insert.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    ASTNodeVariant Insert::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};

        return std::make_shared<Insert>(
            loc, std::visit(cloneVisitor, mKeys),
            mValues | ranges::views::transform([&cloneVisitor](auto &el) { return std::visit(cloneVisitor, el); }));
    }
} // namespace voila::ast