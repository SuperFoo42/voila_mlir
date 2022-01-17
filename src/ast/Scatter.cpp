#include "ast/Scatter.hpp"

namespace voila::ast
{
    Scatter::Scatter(const Location loc, Expression idxs, Expression src_col) :
        IExpression(loc), idxs{std::move(idxs)}, src{std::move(src_col)}
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
        ostream << "scatter( " << src << "," << idxs << ")";
    }
    void Scatter::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Scatter::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast