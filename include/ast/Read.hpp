#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Read : public BinaryOP<Expression>
    {
      public:
        Read(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] bool is_read() const final;

        Read *as_read() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast