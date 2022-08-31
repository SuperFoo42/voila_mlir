#include "ast/Write.hpp"

namespace voila::ast
{
    Write::Write(const Location loc, Expression src_col, Expression dest_col, Expression wpos) :
        IStatement(loc), dest{std::move(dest_col)}, start{std::move(wpos)}, src{std::move(src_col)}
    {
    }

    bool Write::is_write() const
    {
        return true;
    }

    Write *Write::as_write()
    {
        return this;
    }

    std::string Write::type2string() const
    {
        return "write";
    }

    void Write::print(std::ostream &os) const
    {
        os << "src: " << src << " dest: " << dest;
    }
    void Write::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Write::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Write::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_unique<Write>(loc, src.clone(vmap), dest.clone(vmap), start.clone(vmap));
    }
} // namespace voila::ast