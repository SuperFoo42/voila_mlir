#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"
#include "Logical.hpp"
namespace voila::ast
{
    class Or : BinaryOP<Expression>, Logical
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        std::string type2string() const final
        {
            return "or";
        }

        bool is_or() const final
        {
            return true;
        }

        Or *as_or() final
        {
            return this;
        }
    };
} // namespace voila::ast