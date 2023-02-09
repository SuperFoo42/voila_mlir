#include "ast/Write.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/Expression.hpp" // for Expression, operator<<
#include "ast/IStatement.hpp" // for IStatement
#include <ostream>            // for operator<<, ostream, basic_ostream
#include <utility>            // for move
#include <vector>             // for allocator

namespace voila::ast
{
    Write::Write(const Location loc, Expression src_col, Expression dest_col, Expression wpos)
        : IStatement(loc), mDest{std::move(dest_col)}, mStart{std::move(wpos)}, mSrc{std::move(src_col)}
    {
    }

    bool Write::is_write() const { return true; }

    Write *Write::as_write() { return this; }

    std::string Write::type2string() const { return "write"; }

    void Write::print(std::ostream &os) const { os << "src: " << mSrc << " dest: " << mDest; }
    void Write::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Write::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Write::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        return std::make_shared<Write>(loc, mSrc.clone(vmap), mDest.clone(vmap), mStart.clone(vmap));
    }
} // namespace voila::ast