#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrMax : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_max() const final;

        AggrMax *as_aggr_max() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast