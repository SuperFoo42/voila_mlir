#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Eq : public BinaryOP<Expression>, public Comparison
    {
      public:
        Eq(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_eq() const final;

        Eq *as_eq() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast