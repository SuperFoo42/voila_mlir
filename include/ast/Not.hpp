#pragma once
#include "Expression.hpp"
#include "Logical.hpp"
#include "UnaryOP.hpp"
namespace voila::ast
{
    class Not : UnaryOP<Expression>, Logical
    {
      public:
        using UnaryOP::param;
        using UnaryOP::UnaryOP;

        std::string type2string() const final
        {
            return "not";
        }

        bool is_not() const final
        {
            return true;
        }

        Not *as_not() final
        {
            return this;
        }
    };
} // namespace voila::ast