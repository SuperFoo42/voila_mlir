#pragma once
#include "IExpression.hpp"
namespace voila::ast
{
    class Logical : IExpression
    {
        bool is_logical() const final
        {
            return true;
        }

        Logical *as_logical() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "logical";
        }
    };
} // namespace voila::ast