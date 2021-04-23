#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Scatter : public IStatement
    {
      public:
        Scatter(std::string dest_col, Expression idxs, Expression src_col);

        [[nodiscard]] bool is_scatter() const final;

        Scatter *as_scatter() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        std::string dest;
        Expression idxs;
        Expression src;
    };

} // namespace voila::ast