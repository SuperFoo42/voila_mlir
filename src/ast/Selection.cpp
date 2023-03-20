#include "ast/Selection.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    void Selection::print_impl(std::ostream &ostream) const
    {
        auto pVisitor = overloaded{[&ostream](auto &v) { ostream << *v; }, [](std::monostate) {}};
        ostream << type2string() << "( ";
        std::visit(pVisitor, mParam);
        ostream << ",";
        std::visit(pVisitor, mPred);
        ostream << ")";
    }
    ASTNodeVariant Selection::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Selection>(loc, std::visit(cloneVisitor, mParam), std::visit(cloneVisitor, mPred));
    }
} // namespace voila::ast