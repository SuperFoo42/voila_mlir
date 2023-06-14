#include "ast/Load.hpp"
#include "ASTNodes.hpp"
namespace voila::ast
{

    ASTNodeVariant Load::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Load>(loc, std::visit(cloneVisitor, mSrc), std::visit(cloneVisitor, mDest),
                                      std::visit(cloneVisitor, mMask));
    }
} // namespace voila::ast