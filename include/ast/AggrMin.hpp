#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrMin : Aggregation
    {
      public:
        using Aggregation::Aggregation;

        bool is_aggr_min() const final
        {
            return true;
        }

        AggrMin *as_aggr_min() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "min aggregation";
        }
    };
} // namespace voila::ast