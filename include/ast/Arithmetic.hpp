#pragma once
#include "IExpression.hpp"

namespace voila::ast
{
    class Arithmetic : virtual public IExpression
    {
      public:
        ~Arithmetic() override = default;

        [[nodiscard]] bool is_arithmetic() const final;

        Arithmetic *as_arithmetic() final;

        [[nodiscard]] std::string type2string() const override;
    };
} // namespace voila::ast