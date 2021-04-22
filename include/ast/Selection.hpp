#pragma once
#include "Expression.hpp"
#include "UnaryOP.hpp"

namespace voila::ast
{
    class Selection : UnaryOP<Expression>
    {
      public:
        using UnaryOP::param;
        using UnaryOP::UnaryOP;

        bool is_select() const final
        {
            return true;
        }

        Selection *as_select() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "selection";
        }
    };
} // namespace voila::ast