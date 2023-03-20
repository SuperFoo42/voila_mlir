#include "ast/Not.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    ASTNodeVariant Not::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        return std::make_shared<Not>(loc, std::visit(overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                                                [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }}, mParam));
    }
} // namespace voila::ast