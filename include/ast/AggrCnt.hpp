#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class AggrCnt : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_cnt() const final;

        AggrCnt *as_aggr_cnt() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast