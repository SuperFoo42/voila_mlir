#include "ast/Or.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    std::string Or::type2string() const { return "or"; }
    bool Or::is_or() const { return true; }
    Or *Or::as_or() { return this; }
    ASTNodeVariant Or::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Or>(loc, std::visit(cloneVisitor,mLhs), std::visit(cloneVisitor,mRhs));
    }
} // namespace voila::ast