#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrAvg : Aggregation
    {
      public:
        using Aggregation::Aggregation;

        bool is_aggr_avg() const final
        {
            return true;
        }

        AggrAvg *as_aggr_avg() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "avg aggregation";
        }
    };
} // namespace voila::ast