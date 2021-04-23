#include "Expression.hpp"
#include "BinaryOP.hpp"
#include "Arithmetic.hpp"

#pragma once

namespace voila::ast
{
    class Sub : public BinaryOP<Expression>, public Arithmetic
    {
      public:
        using BinaryOP::BinaryOP;
        using BinaryOP::lhs;
        using BinaryOP::rhs;

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_sub() const final;
        void print(std::ostream &ostream) const final;

      protected:
        void checkArgs(Expression &lhs, Expression &rhs) final;
    };
}

