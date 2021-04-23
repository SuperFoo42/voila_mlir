#pragma once
#include "Arithmetic.hpp"
#include "BinaryOP.hpp"
#include "Expression.hpp"

namespace voila::ast
{
    // Expressions
    class Add : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_add() const final;

        Add *as_add() final;
        void print(std::ostream &ostream) const final;

      protected:
        void checkArgs(Expression &lhs, Expression &rhs) final;
    };
} // namespace voila::ast