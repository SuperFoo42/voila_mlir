#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Ge : public BinaryOP<Expression>, public Comparison
    {
      public:
        Ge(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_ge() const final;

        Ge *as_ge() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast