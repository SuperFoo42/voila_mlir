#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    class Arithmetic : IExpression
    {
      public:
        virtual ~Arithmetic() = default;

        bool is_arithmetic() const final
        {
            return true;
        }

        Arithmetic *as_arithmetic() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "arithmetic";
        }
    };
} // namespace voila::ast