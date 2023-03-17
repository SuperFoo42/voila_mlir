#include "ast/Gather.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    bool Gather::is_gather() const { return true; }
    Gather *Gather::as_gather() { return this; }
    std::string Gather::type2string() const { return "gather"; }
    void Gather::print(std::ostream &ostream) const
    {
        auto pVisitor = overloaded{[&ostream](auto &v) { ostream << *v; }, [](std::monostate) {}};
        ostream << type2string() << "( ";
        std::visit(pVisitor, mColumn);
        ostream << ",";
        std::visit(pVisitor, mIdxs);
        ostream << ")";
    }

    ASTNodeVariant Gather::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Gather>(loc, std::visit(cloneVisitor, mColumn), std::visit(cloneVisitor, mIdxs));
    }
} // namespace voila::ast