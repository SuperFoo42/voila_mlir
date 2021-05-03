#include "ast/Scatter.hpp"

namespace voila::ast
{
    Scatter::Scatter(const Location loc, Expression src_col, Expression dest_col, Expression idxs) :
        IStatement(loc), dest{std::move(dest_col)}, idxs{std::move(idxs)}, src{std::move(src_col)}
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
        ostream << "dest: " << dest;
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