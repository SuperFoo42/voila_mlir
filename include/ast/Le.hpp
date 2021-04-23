#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Le : public BinaryOP<Expression>, public Comparison
    {
      public:
        Le(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_le() const final;

        Le *as_le() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast