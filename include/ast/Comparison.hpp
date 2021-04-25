#pragma once
#include <utility>

#include "IExpression.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Comparison : virtual public IExpression
    {
      public:
        Comparison(Expression lhs, Expression rhs) : lhs{std::move(lhs)}, rhs{std::move(rhs)} {}
        [[nodiscard]] bool is_comparison() const final;

        Comparison *as_comparison() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &ostream) const final;

        Expression lhs;
        Expression rhs;
    };
} // namespace voila::ast