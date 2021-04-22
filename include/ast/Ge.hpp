#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Ge : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "ge";
        }

        bool is_ge() const final
        {
            return true;
        }

        Ge * as_ge() final
        {
            return this;
        }
    };
} // namespace voila::ast