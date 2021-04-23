#pragma once
#include "Expression.hpp"
#include "UnaryOP.hpp"

namespace voila::ast
{
    class Selection : public UnaryOP<Expression>
    {
      public:
        using UnaryOP::param;
        explicit Selection(Expression expr) : UnaryOP<Expression>(expr)
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
    };
} // namespace voila::ast