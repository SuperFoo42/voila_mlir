#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    class Comparison : IExpression
    {
        bool is_comparison() const final
        {
            return true;
        }

        Comparison *as_comparison() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "comparison";
        }
    };
} // namespace voila::ast