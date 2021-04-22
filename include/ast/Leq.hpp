#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Leq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "leq";
        }

        bool is_leq() const final
        {
            return true;
        }

        Leq *as_leq() final
        {
            return this;
        }
    };
} // namespace voila::ast