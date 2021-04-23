#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrAvg : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_avg() const final;

        AggrAvg *as_aggr_avg() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast