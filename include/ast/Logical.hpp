#pragma once
#include "IExpression.hpp"
namespace voila::ast
{
    class Logical : public IExpression
    {
      public:
        explicit Logical(const Location loc) : IExpression(loc){};
        ~Logical() override = default;
        [[nodiscard]] bool is_logical() const final;

        Logical *as_logical() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &) const final;
    };
} // namespace voila::ast