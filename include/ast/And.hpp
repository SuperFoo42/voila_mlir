#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"
#include "Logical.hpp"
namespace voila::ast
{
    class And : BinaryOP<Expression>, Logical
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "and";
        }

        bool is_and() const final
        {
            return true;
        }

        And *as_and() final
        {
            return this;
        }
    };
} // namespace voila::ast