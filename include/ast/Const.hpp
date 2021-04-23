#pragma once
#include "IExpression.hpp"
namespace voila::ast
{
    class Const : virtual public IExpression
    {
      public:
        ~Const() override = default;

        [[nodiscard]] bool is_const() const final;

        Const *as_const() final;

        [[nodiscard]] std::string type2string() const override;
    };
} // namespace voila::ast