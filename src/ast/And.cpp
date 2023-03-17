#include "ast/And.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string And::type2string() const
    {
        return "and";
    }

    bool And::is_and() const
    {
        return true;
    }

    And *And::as_and()
    {
        return this;
    }

    ASTNodeVariant And::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &)-> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<And>(loc, std::visit(cloneVisitor, mLhs), std::visit(cloneVisitor, mRhs));
    }
} // namespace voila::ast