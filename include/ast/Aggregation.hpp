#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>

namespace voila::ast
{
    class Aggregation : public IStatement
    {
      public:
        Aggregation(const Location loc, Expression col) : IStatement(loc), src{std::move(col)} {}
        ~Aggregation() override = default;

        [[nodiscard]] bool is_aggr() const final;

        Aggregation *as_aggr() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &) const final {}

        Expression src;
    };
} // namespace voila::ast