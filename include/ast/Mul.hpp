#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Mul : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "mul";
        }

        bool is_mul() const final
        {
            return true;
        }

        Mul *as_mul() final
        {
            return this;
        }
    };
} // namespace voila::ast