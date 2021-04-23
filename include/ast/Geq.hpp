#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Geq : public BinaryOP<Expression>, public Comparison
    {
      public:
        Geq(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_geq() const final;

        Geq *as_geq() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast