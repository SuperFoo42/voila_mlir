#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>

namespace voila::ast
{
    class Aggregation : virtual public IStatement
    {
      public:
        Aggregation(Expression col, Expression idxs) : IStatement(), src{col}, idxs{idxs} {}
        ~Aggregation() override = default;

        [[nodiscard]] bool is_aggr() const final;

        Aggregation *as_aggr() final;

        [[nodiscard]] std::string type2string() const override;

        Expression src;
        Expression idxs;
    };
} // namespace voila::ast