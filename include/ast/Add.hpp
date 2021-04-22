#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    // Expressions
    class Add : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "add";
        }

        bool is_add() const final
        {
            return true;
        }

        Add *as_add() final
        {
            return this;
        }
    };
} // namespace voila::ast