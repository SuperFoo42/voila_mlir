#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Mod : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        Mod(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_mod() const final;

        Mod *as_mod() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast