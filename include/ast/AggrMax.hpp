#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrMax : Aggregation
    {
      public:
        using Aggregation::Aggregation;

        bool is_aggr_max() const final
        {
            return true;
        }

        AggrMax *as_aggr_max() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "max aggregation";
        }
    };
} // namespace voila::ast