#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Geq : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "geq";
        }

        bool is_geq() const final
        {
            return true;
        }

        Geq *as_geq() final
        {
            return this;
        }
    };
} // namespace voila::ast