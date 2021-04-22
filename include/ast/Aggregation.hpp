#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

namespace voila::ast
{
    class Aggregation : IStatement
    {
      public:
        Aggregation(const Expression &col, const Expression &idxs) : IStatement(), src{col}, idxs{idxs} {}
        virtual ~Aggregation() = default;

        bool is_aggr() const final
        {
            return true;
        }

        Aggregation *as_aggr() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "aggregation";
        }

        Expression src;
        Expression idxs;
    };
} // namespace voila::ast