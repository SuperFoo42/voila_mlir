#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrSum : Aggregation
    {
      public:
        using Aggregation::Aggregation;

        bool is_aggr_sum() const final
        {
            return true;
        }

        AggrSum *as_aggr_sum() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "sum aggregation";
        }
    };
} // namespace voila::ast