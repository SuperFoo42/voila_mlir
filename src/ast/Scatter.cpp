#include "ast/Scatter.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include <ostream>            // for operator<<, ostream, basic_ostream
#include <vector>             // for allocator

namespace voila::ast
{
    void Scatter::print_impl(std::ostream &ostream) const
    {
        auto pVisitor = overloaded{[&ostream](auto &v) { ostream << *v; }, [](std::monostate) {}};
        ostream << type2string() << "( ";
        std::visit(pVisitor, mSrc);
        ostream << ",";
        std::visit(pVisitor, mIdxs);
        ostream << ")";
    }
    ASTNodeVariant Scatter::clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Scatter>(loc, std::visit(cloneVisitor, mIdxs), std::visit(cloneVisitor, mSrc));
    }

} // namespace voila::ast