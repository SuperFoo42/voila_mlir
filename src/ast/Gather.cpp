#include "ast/Gather.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor

namespace voila::ast
{
    void Gather::print_impl(std::ostream &ostream) const
    {
        auto pVisitor = overloaded{[&ostream](auto &v) -> void { ostream << *v; }, [](std::monostate) {}};
        ostream << type2string_impl() << "( ";
        std::visit(pVisitor, mColumn);
        ostream << ",";
        std::visit(pVisitor, mIdxs);
        ostream << ")";
    }

    ASTNodeVariant Gather::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Gather>(loc, std::visit(cloneVisitor, mColumn), std::visit(cloneVisitor, mIdxs));
    }
} // namespace voila::ast