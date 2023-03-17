#include "ast/Not.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    std::string Not::type2string() const { return "not"; }
    bool Not::is_not() const { return true; }
    Not *Not::as_not() { return this; }

    ASTNodeVariant Not::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Not>(loc, std::visit(cloneVisitor, mParam));
    }
} // namespace voila::ast