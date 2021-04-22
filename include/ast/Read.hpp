#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Read : BinaryOP<Expression>
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        bool is_read() const final
        {
            return true;
        }

        Read *as_read() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "read";
        }
    };
} // namespace voila::ast