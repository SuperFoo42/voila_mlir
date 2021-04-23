#pragma once
#include "BinaryOP.hpp"
#include "Expression.hpp"
#include "Logical.hpp"
namespace voila::ast
{
    class Or : public BinaryOP<Expression>, public Logical
    {
      public:
        Or(Expression lhs, Expression rhs) : BinaryOP<Expression>(lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_or() const final;

        Or *as_or() final;
        void print(std::ostream &ostream) const final;
    };
} // namespace voila::ast