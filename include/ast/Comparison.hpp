#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    class Comparison : virtual public IExpression
    {
        [[nodiscard]] bool is_comparison() const final;

        Comparison *as_comparison() final;

        [[nodiscard]] std::string type2string() const override;
    };
} // namespace voila::ast