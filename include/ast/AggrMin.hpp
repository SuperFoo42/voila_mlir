#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrMin : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_min() const final;

        AggrMin *as_aggr_min() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast