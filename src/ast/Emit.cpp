#include "ast/Emit.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include <algorithm>          // for max
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    bool Emit::is_emit() const { return true; }

    Emit *Emit::as_emit() { return this; }

    std::string Emit::type2string() const { return "emit"; }

    void Emit::print(std::ostream &) const {}

    ASTNodeVariant Emit::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedExprs;
        for (auto &arg : mExprs)
        {
            auto cloneVisitor =
                overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                           [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
            clonedExprs.push_back(std::visit(cloneVisitor, arg));
        }

        return std::make_shared<Emit>(loc, clonedExprs);
    }
} // namespace voila::ast