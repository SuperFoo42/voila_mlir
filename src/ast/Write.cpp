#include "ast/Write.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/IStatement.hpp" // for IStatement
#include <ostream>            // for operator<<, ostream, basic_ostream
#include <utility>            // for move
#include <vector>             // for allocator

namespace voila::ast
{
    Write::Write(const Location loc, ASTNodeVariant src_col, ASTNodeVariant dest_col, ASTNodeVariant wpos)
        : IStatement(loc), mDest{std::move(dest_col)}, mStart{std::move(wpos)}, mSrc{std::move(src_col)}
    {
    }

    bool Write::is_write() const { return true; }

    Write *Write::as_write() { return this; }

    std::string Write::type2string() const { return "write"; }

    void Write::print(std::ostream &os) const
    {
        std::visit(overloaded{[&os](auto &src) { os << "src: " << *src; }, [](std::monostate) {}}, mSrc);
        std::visit(overloaded{[&os](auto &dest) { os << "dest: " << *dest; }, [](std::monostate) {}}, mDest);
    }

    ASTNodeVariant Write::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<Write>(loc, std::visit(cloneVisitor, mSrc), std::visit(cloneVisitor, mDest), std::visit(cloneVisitor, mStart));
    }
    const ASTNodeVariant &Write::dest() const { return mDest; }
    const ASTNodeVariant &Write::start() const { return mStart; }
    const ASTNodeVariant &Write::src() const { return mSrc; }
} // namespace voila::ast