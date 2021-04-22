#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Div : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "div";
        }

        bool is_div() const final
        {
            return true;
        }

        Div *as_div() final
        {
            return this;
        }
    };
} // namespace voila::ast