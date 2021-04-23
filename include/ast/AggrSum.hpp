#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrSum : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_sum() const final;

        AggrSum *as_aggr_sum() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast