#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class Scatter : IStatement
    {
      public:
        Scatter(const Expression &dest_col, const Expression &idxs, const Expression &src_col) :
            IStatement(), dest{dest_col}, idxs{idxs}, src{src_col}
        {
        }

        bool is_scatter() const final
        {
            return true;
        }

        Scatter *as_scatter() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "scatter";
        }

        Expression dest;
        Expression idxs;
        Expression src;
    };

} // namespace voila::ast