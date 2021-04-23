#include "ast/Scatter.hpp"

namespace voila::ast
{
    Scatter::Scatter(std::string dest_col, Expression idxs, Expression src_col) :
        IStatement(), dest{std::move(dest_col)}, idxs{std::move(idxs)}, src{std::move(src_col)}
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
        ostream << "scatter";
    }
} // namespace voila::ast