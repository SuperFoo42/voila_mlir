#include "ast/Predicate.hpp"
#include "ASTNodes.hpp"
namespace voila::ast
{

    ASTNodeVariant voila::ast::Predicate::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        return std::make_shared<Predicate>(
            loc, std::visit(overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }},
                            mExpr));
    }
} // namespace voila::ast