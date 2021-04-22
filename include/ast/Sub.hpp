#include "Expression.hpp"
#include "BinaryOP.hpp"
#include "Arithmetic.hpp"

#pragma once

namespace voila::ast
{
    class Sub : BinaryOP<Expression>, Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "sub";
        }

        bool is_sub() const final
        {
            return true;
        }
    };
}

