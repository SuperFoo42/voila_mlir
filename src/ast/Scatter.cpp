#include "ast/Scatter.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include <ostream>            // for operator<<, ostream, basic_ostream
#include <utility>            // for move
#include <vector>             // for allocator

namespace voila::ast
{
    Scatter::Scatter(const Location loc, ASTNodeVariant idxs, ASTNodeVariant src_col)
        : IExpression(loc), mIdxs{std::move(idxs)}, mSrc{std::move(src_col)}
    {
    }
    bool Scatter::is_scatter() const { return true; }
    Scatter *Scatter::as_scatter() { return this; }
    std::string Scatter::type2string() const { return "scatter"; }

    void Scatter::print(std::ostream &ostream) const
    {
        auto pVisitor = overloaded{[&ostream](auto &v) { ostream << *v; }, [](std::monostate) {}};
        ostream << type2string() << "( ";
        std::visit(pVisitor, mSrc);
        ostream << ",";
        std::visit(pVisitor, mIdxs);
        ostream << ")";
    }

    ASTNodeVariant Scatter::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Scatter>(loc, std::visit(cloneVisitor, mIdxs), std::visit(cloneVisitor, mSrc));
    }
} // namespace voila::ast