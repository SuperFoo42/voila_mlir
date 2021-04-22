#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Neq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;
        std::string type2string() const final
        {
            return "neq";
        }

        bool is_neq() const final
        {
            return true;
        }

        Neq *as_neq() final
        {
            return this;
        }
    };
} // namespace voila::ast