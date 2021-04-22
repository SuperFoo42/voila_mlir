#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Eq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "eq";
        }

        bool is_eq() const final
        {
            return true;
        }

        Eq *as_eq() final
        {
            return this;
        }
    };
} // namespace voila::ast