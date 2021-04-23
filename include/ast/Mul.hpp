#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Mul : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        Mul(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_mul() const final;

        Mul *as_mul() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast