#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Div : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        Div(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_div() const final;

        Div *as_div() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast