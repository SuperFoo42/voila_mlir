#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Leq : public BinaryOP<Expression>, public Comparison
    {
      public:
        Leq(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_leq() const final;

        Leq *as_leq() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast