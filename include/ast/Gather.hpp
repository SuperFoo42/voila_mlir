#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Gather : BinaryOP<Expression>
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        bool is_gather() const final
        {
            return true;
        }

        Gather *as_gather() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "gather";
        }
    };
} // namespace voila::ast