#include "ast/StatementWrapper.hpp"
#include "ASTNodes.hpp"

namespace voila::ast
{
    ASTNodeVariant StatementWrapper::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        return std::make_shared<StatementWrapper>(
            loc, std::visit(overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }},
                            mExpr));
    }
} // namespace voila::ast