#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrCnt : Aggregation
    {
      public:
        using Aggregation::Aggregation;

        bool is_aggr_cnt() const final
        {
            return true;
        }

        AggrCnt *as_aggr_cnt() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "count aggregation";
        }
    };
} // namespace voila::ast