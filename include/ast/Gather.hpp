#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Gather : public BinaryOP<Expression>
    {
      public:
        Gather(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] bool is_gather() const final;

        Gather *as_gather() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast