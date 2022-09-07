#include "ast/Scatter.hpp"

namespace voila::ast
{
    Scatter::Scatter(const Location loc, Expression idxs, Expression src_col) :
        IExpression(loc), mIdxs{std::move(idxs)}, mSrc{std::move(src_col)}
    {
    }
    bool Scatter::is_scatter() const
    {
        return true;
    }
    Scatter *Scatter::as_scatter()
    {
        return this;
    }
    std::string Scatter::type2string() const
    {
        return "scatter";
    }
    void Scatter::print(std::ostream &ostream) const
    {
        ostream << "scatter( " << mSrc << "," << mIdxs << ")";
    }
    void Scatter::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Scatter::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> Scatter::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_shared<Scatter>(loc, mIdxs.clone(vmap), mSrc.clone(vmap));
    }
} // namespace voila::ast