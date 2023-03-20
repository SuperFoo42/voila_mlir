#include "ast/Lookup.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{

    ASTNodeVariant Lookup::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        auto cloneVisitTransformer = ranges::views::transform([&cloneVisitor](auto &e) {return std::visit(cloneVisitor, e);});

        return std::make_shared<Lookup>(loc, mValues | cloneVisitTransformer, mTables | cloneVisitTransformer, std::visit(cloneVisitor, mHashes));
    }
} // namespace voila::ast