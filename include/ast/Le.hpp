#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Le : BinaryOP<Expression>, Comparison
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "le";
        }

        bool is_le() const final
        {
            return true;
        }

        Le *as_le() final
        {
            return this;
        }
    };
} // namespace voila::ast