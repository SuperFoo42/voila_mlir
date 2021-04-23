#include "Expression.hpp"
#include "BinaryOP.hpp"
#include "Arithmetic.hpp"

#pragma once

namespace voila::ast
{
    class Sub : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        Sub(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_sub() const final;
        void print(std::ostream &ostream) const final;
    };
}

