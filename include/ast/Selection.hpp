#pragma once
#include "Expression.hpp"
#include "UnaryOP.hpp"

namespace voila::ast
{
    class Selection : public UnaryOP<Expression>
    {
      public:
        using UnaryOP::param;
        using UnaryOP::UnaryOP;

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;

        void checkArg(const Expression &) override;
    };
} // namespace voila::ast