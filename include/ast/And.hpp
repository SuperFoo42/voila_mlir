#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"
#include "Logical.hpp"
namespace voila::ast
{
    class And : public BinaryOP<Expression>, public Logical
    {
      public:
        And(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_and() const final;

        And *as_and() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast