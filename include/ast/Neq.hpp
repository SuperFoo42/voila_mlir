#pragma once
#include "BinaryOP.hpp"
#include "Comparison.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    class Neq : public BinaryOP<Expression>, public Comparison
    {
      public:
        Neq(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        { // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_neq() const final;

        Neq *as_neq() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast